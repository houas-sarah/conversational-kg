"""PersonaChat-style evaluation harness.

Measures four metrics on the curated sample in personachat_sample.json:

  1. Extraction Precision  — of triples the system produced, how many match a
                             ground-truth annotation? (high = few hallucinations)
  2. Extraction Recall     — of ground-truth triples, how many did the system
                             find? (high = thorough)
  3. Memory@1              — after ingesting a dialogue, can the retrieval layer
                             surface the expected fact in its top result?
  4. Conflict Accuracy     — for dialogues with temporal conflicts, did the
                             system supersede the right number of facts and
                             keep the right one active?

Run:
    .venv\\Scripts\\python.exe -m eval.run_eval
    .venv\\Scripts\\python.exe -m eval.run_eval --no-llm    # rules-only baseline

Outputs:
    eval/results.csv       per-dialogue numbers
    eval/results.md        human-readable summary you can paste into your report
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from backend.extractor import HybridExtractor
from backend.graph import KnowledgeGraph
from backend.llm import GroqClient
from backend.query import retrieve


def _norm_triple(t) -> tuple[str, str, str]:
    s, p, o = t
    return (s.strip().lower(), p.strip().lower().replace(" ", "_"), o.strip().lower())


def _triple_match(predicted, gold) -> bool:
    ps, pp, po = predicted
    gs, gp, go = gold
    if ps != gs:
        return False
    if pp != gp:
        return False
    return go in po or po in go


async def evaluate_dialogue(dlg: dict, extractor: HybridExtractor) -> dict:
    """Run one dialogue end-to-end and return per-dialogue metrics."""
    tmp_path = Path(ROOT) / "eval" / f"_tmp_{dlg['dialogue_id']}.json"
    kg = KnowledgeGraph(tmp_path)
    try:
        gold_total = 0
        pred_total = 0
        pred_correct = 0
        gold_recovered = 0
        per_turn = []
        latencies: list[float] = []

        for turn in dlg["turns"]:
            text = turn["text"]
            gold_triples = [_norm_triple(t) for t in turn.get("ground_truth", [])]

            t0 = time.time()
            ext = await extractor.extract(text)
            elapsed = time.time() - t0
            latencies.append(elapsed)
            pred_triples = [_norm_triple((t.subject, t.predicate, t.object)) for t in ext.triples]

            for t in ext.triples:
                kg.add_fact(t.subject, t.predicate, t.object, t.confidence)

            gold_total += len(gold_triples)
            pred_total += len(pred_triples)

            matched_pred = set()
            matched_gold = set()
            for gi, gt in enumerate(gold_triples):
                for pi, pt in enumerate(pred_triples):
                    if pi in matched_pred:
                        continue
                    if _triple_match(pt, gt):
                        matched_pred.add(pi)
                        matched_gold.add(gi)
                        break
            pred_correct += len(matched_pred)
            gold_recovered += len(matched_gold)

            per_turn.append({
                "text": text,
                "gold": gold_triples,
                "pred": pred_triples,
                "matched": len(matched_gold),
            })

        memory_hits = 0
        memory_total = 0
        for probe in dlg.get("memory_probes", []):
            ctx = retrieve(kg, probe["question"], k=5)
            top_facts = [(f.subject.lower(), f.predicate.lower(), f.object.lower()) for f in ctx.facts[:3]]
            target_p = probe["expected_predicate"].lower()
            target_o = probe["expected_object"].lower()
            if any(p == target_p and (target_o in o or o in target_o) for _, p, o in top_facts):
                memory_hits += 1
            memory_total += 1

        conflict_pass = None
        cc = dlg.get("conflict_check")
        if cc:
            superseded_count = sum(1 for f in kg.facts.values()
                                   if f.predicate == cc["predicate"] and f.superseded_by is not None)
            active = [f for f in kg.active_facts() if f.predicate == cc["predicate"]]
            final_object_match = (
                len(active) == 1
                and cc["final_object"].lower() in active[0].object.lower()
            )
            count_match = superseded_count == cc["should_supersede_count"]
            conflict_pass = final_object_match and count_match

        return {
            "dialogue_id": dlg["dialogue_id"],
            "description": dlg["description"],
            "gold_total": gold_total,
            "pred_total": pred_total,
            "pred_correct": pred_correct,
            "gold_recovered": gold_recovered,
            "precision": pred_correct / pred_total if pred_total else 0.0,
            "recall": gold_recovered / gold_total if gold_total else 0.0,
            "memory_hits": memory_hits,
            "memory_total": memory_total,
            "memory_at_1": memory_hits / memory_total if memory_total else None,
            "conflict_pass": conflict_pass,
            "avg_latency_s": statistics.mean(latencies) if latencies else 0.0,
            "per_turn": per_turn,
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


async def run(no_llm: bool = False) -> dict:
    sample_path = ROOT / "eval" / "personachat_sample.json"
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    dialogues = data["dialogues"]

    if no_llm:
        os.environ["GROQ_API_KEY"] = ""
    extractor = HybridExtractor(GroqClient())
    method = "hybrid (spaCy + Groq LLM)" if extractor.llm.available else "rules + spaCy (no LLM)"

    print("=" * 70)
    print(f"PERSONACHAT-STYLE EVALUATION  ·  method: {method}")
    print(f"corpus: {len(dialogues)} dialogues, "
          f"{sum(len(d['turns']) for d in dialogues)} user turns")
    print("=" * 70)

    results = []
    for dlg in dialogues:
        r = await evaluate_dialogue(dlg, extractor)
        results.append(r)
        mark = "✓" if (r["recall"] >= 0.5) else "·"
        print(
            f"  {mark} {r['dialogue_id']:<26}"
            f" P={r['precision']:.2f}  R={r['recall']:.2f}"
            f"  mem@1={r['memory_at_1'] if r['memory_at_1'] is not None else '—'}"
            f"  conflict={r['conflict_pass'] if r['conflict_pass'] is not None else '—'}"
            f"  t={r['avg_latency_s']*1000:.0f}ms"
        )

    total_gold = sum(r["gold_total"] for r in results)
    total_pred = sum(r["pred_total"] for r in results)
    total_correct = sum(r["pred_correct"] for r in results)
    total_recovered = sum(r["gold_recovered"] for r in results)
    total_mem_hits = sum(r["memory_hits"] for r in results)
    total_mem = sum(r["memory_total"] for r in results)
    conflict_results = [r["conflict_pass"] for r in results if r["conflict_pass"] is not None]

    micro_p = total_correct / total_pred if total_pred else 0.0
    micro_r = total_recovered / total_gold if total_gold else 0.0
    micro_f1 = f1(micro_p, micro_r)
    macro_p = statistics.mean([r["precision"] for r in results if r["pred_total"]])
    macro_r = statistics.mean([r["recall"] for r in results if r["gold_total"]])
    mem_acc = total_mem_hits / total_mem if total_mem else 0.0
    conflict_acc = sum(conflict_results) / len(conflict_results) if conflict_results else None
    avg_latency = statistics.mean([r["avg_latency_s"] for r in results]) * 1000

    summary = {
        "method": method,
        "dialogues": len(dialogues),
        "turns_total": sum(len(d["turns"]) for d in dialogues),
        "extraction": {
            "gold_total": total_gold,
            "pred_total": total_pred,
            "pred_correct": total_correct,
            "gold_recovered": total_recovered,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
        },
        "memory_at_1": mem_acc,
        "memory_probes_total": total_mem,
        "conflict_accuracy": conflict_acc,
        "conflict_cases": len(conflict_results),
        "avg_latency_ms": avg_latency,
    }

    print()
    print("─" * 70)
    print("OVERALL")
    print("─" * 70)
    print(f"  Extraction micro-precision  {micro_p:.3f}   (PDF §8 target ≥ 0.90)")
    print(f"  Extraction micro-recall     {micro_r:.3f}")
    print(f"  Extraction micro-F1         {micro_f1:.3f}")
    print(f"  Memory@1 accuracy           {mem_acc:.3f}   (PDF §8 target ≥ 0.85)")
    if conflict_acc is not None:
        print(f"  Conflict resolution accuracy {conflict_acc:.3f}   ({sum(conflict_results)}/{len(conflict_results)} cases)")
    print(f"  Avg per-turn latency        {avg_latency:.0f} ms   (PDF §8 target < 2000)")

    suffix = "_rules" if no_llm else "_hybrid"
    csv_path = ROOT / "eval" / f"results{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["dialogue_id", "gold", "pred", "correct", "precision", "recall",
                    "memory_at_1", "conflict_pass", "avg_latency_ms"])
        for r in results:
            w.writerow([
                r["dialogue_id"], r["gold_total"], r["pred_total"], r["pred_correct"],
                f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                "" if r["memory_at_1"] is None else f"{r['memory_at_1']:.3f}",
                "" if r["conflict_pass"] is None else str(r["conflict_pass"]),
                f"{r['avg_latency_s']*1000:.0f}",
            ])

    md_path = ROOT / "eval" / f"results{suffix}.md"
    lines = [
        "# PersonaChat-style Evaluation Report",
        "",
        f"_{summary['method']}_  ·  {summary['dialogues']} dialogues  ·  {summary['turns_total']} user turns",
        "",
        "## Headline numbers",
        "",
        "| Metric | Result | PDF §8 target |",
        "|---|---|---|",
        f"| Extraction precision (micro) | **{micro_p:.1%}** | ≥ 90% |",
        f"| Extraction recall (micro)    | **{micro_r:.1%}** | — |",
        f"| Extraction F1 (micro)        | **{micro_f1:.1%}** | — |",
        f"| Memory@1 accuracy            | **{mem_acc:.1%}** | ≥ 85% |",
        (f"| Conflict resolution accuracy | **{conflict_acc:.1%}** ({sum(conflict_results)}/{len(conflict_results)}) | — |"
         if conflict_acc is not None else "| Conflict resolution accuracy | — | — |"),
        f"| Avg per-turn latency         | **{avg_latency:.0f} ms** | < 2000 ms |",
        "",
        "## Per-dialogue breakdown",
        "",
        "| Dialogue | P | R | mem@1 | conflict | latency |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        mem = "—" if r["memory_at_1"] is None else f"{r['memory_at_1']:.2f}"
        conf = "—" if r["conflict_pass"] is None else ("✓" if r["conflict_pass"] else "✗")
        lines.append(
            f"| `{r['dialogue_id']}` "
            f"| {r['precision']:.2f} | {r['recall']:.2f} "
            f"| {mem} | {conf} | {r['avg_latency_s']*1000:.0f} ms |"
        )
    lines.append("")
    lines.append("Generated by `eval/run_eval.py`. Re-run to refresh.")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"  wrote {csv_path.relative_to(ROOT)}")
    print(f"  wrote {md_path.relative_to(ROOT)}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="rules-only baseline (disables Groq)")
    args = parser.parse_args()
    asyncio.run(run(no_llm=args.no_llm))


if __name__ == "__main__":
    main()
