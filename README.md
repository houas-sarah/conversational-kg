<div align="center">

# the commonplace

***a notebook that listens, and remembers what was said.***

A conversational AI built on a **time-aware knowledge graph**. Every message
you send is parsed into triples, stamped with a timestamp, and merged into a
graph that grows and corrects itself across turns. The graph is the chatbot's
long-term memory — and you can watch it form in real time.

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![D3.js](https://img.shields.io/badge/D3.js-F9A03C.svg?logo=d3.js&logoColor=white)](https://d3js.org/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.1-f55036.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[**what it does**](#what-it-does) · [**run it**](#run-it) · [**how it works**](#how-it-works) · [**evaluation**](#evaluation) · [**deploy**](#deploy)

</div>

---

## what it does

Most chatbots forget. Tell them you live in Algiers on Monday, and by Friday
they've never heard of the place. This one keeps a graph.

```
you   ›  "my name is sarah, i'm studying calculus"
       ↓
graph ›  [user] —named→ [sarah]
         [user] —studies→ [calculus]
       ↓
you   ›  (three days later) "i'm switching to physics"
       ↓
graph ›  [user] —studies→ [physics]
         (previous "studies calculus" marked superseded, not deleted)
       ↓
you   ›  "what do you remember about me?"
       ↓
reply ›  "your name's sarah and you switched to physics — how's it going?"
```

The graph keeps everything: when each fact was learned, what it superseded, and
how confident the system was. The chatbot uses it as a retrieval layer before
generating replies — so it can't claim to know things it never heard.

### features

- **hybrid extraction** — spaCy NER + Groq's Llama 3.1 for relations, with a regex fallback if no API key is set
- **time-aware graph** — every triple is timestamped; 1:1 facts (`lives_in`, `studies`) get superseded by newer ones; multi-value facts (`likes`, `feels`) accumulate
- **conflict resolution** — `superseded_by` pointers preserve the timeline; opposite-polarity facts collide (coming to *understand* what you *struggled with* closes the old one), a repeated fact reinforces instead of duplicating, and the UI shows corrections live
- **per-session memory** — each visitor gets an isolated graph keyed by a browser id, with idle eviction and a session cap, so the live demo is safely multi-user
- **live visualisation** — D3 force-directed graph that updates over a WebSocket as you chat
- **retrieval-grounded replies** — the responder is handed the top-k relevant facts before generating, so the chatbot won't invent things
- **51 pytest tests** — covers thread-safety, persistence round-trip, corrupted-input recovery, chained supersessions, reinforcement and polarity conflicts
- **12-dialogue evaluation harness** — produces real precision / recall / memory@1 / latency numbers (see below)

---

## run it

### prerequisites

- Python 3.12+
- Optional but recommended: a free [Groq](https://console.groq.com) API key (no credit card needed) for hybrid LLM extraction. Without it the system still works using rules+spaCy.

### windows (PowerShell)

```powershell
git clone https://github.com/<you>/the-commonplace.git
cd the-commonplace
.\run.ps1 -Setup     # one-time: venv, pip install, spaCy model
.\run.ps1            # start the server
```

### macOS / linux

```bash
git clone https://github.com/<you>/the-commonplace.git
cd the-commonplace
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env  # add your GROQ_API_KEY (optional)
uvicorn backend.main:app --reload --port 8000
```

Open <http://localhost:8000>.

### pre-loaded demo data

```bash
python -m backend.seed_data --reset    # 4 multi-day conversations, ~20 nodes
```

---

## how it works

Five layers, each isolated behind a small interface:

```
                       ┌─────────────────────────────────────┐
   user message  →     │  spaCy NER  +  Groq Llama 3.1       │   extraction
                       │  → list of (subject, pred, object)  │
                       └─────────────────┬───────────────────┘
                                         ↓
                       ┌─────────────────────────────────────┐
                       │  NetworkX MultiDiGraph              │   storage
                       │  + timestamps                        │
                       │  + superseded_by pointers            │
                       │  + thread-safe upsert                │
                       └─────────────────┬───────────────────┘
                                         ↓
                       ┌─────────────────────────────────────┐
                       │  token-overlap retrieval            │   query
                       │  + 2-hop graph walk                  │
                       └─────────────────┬───────────────────┘
                                         ↓
                       ┌─────────────────────────────────────┐
                       │  Groq Llama 3.1 grounded reply      │   responder
                       │  (template fallback)                 │
                       └─────────────────┬───────────────────┘
                                         ↓
                       ┌─────────────────────────────────────┐
                       │  FastAPI + WebSocket                 │   live UI
                       │  D3 force-directed graph             │
                       └─────────────────────────────────────┘
```

### why this stack

The original spec proposed Neo4j + spaCy + SPARQL. This implementation chose:

| Original | Replaced with | Why |
|---|---|---|
| Neo4j (Docker server, ~1 GB) | NetworkX + JSON | Zero install, runs in-process. Same property-graph model; swapping to Neo4j later is one file. |
| spaCy-only extraction | spaCy + Groq Llama 3.1 (hybrid) | The proposal's own comparison shows hybrid/LLM wins on precision. Free tier on Groq, no card. |
| SPARQL | Cypher-style traversal in code | SPARQL targets RDF triple stores. We're in property-graph world. |

---

## evaluation

A 12-dialogue PersonaChat-style sample is in [`eval/personachat_sample.json`](eval/personachat_sample.json), with ground-truth triples and memory probes. Run both methods:

```bash
python -m eval.run_eval            # hybrid
python -m eval.run_eval --no-llm   # rules-only baseline
```

Outputs are saved as `eval/results_hybrid.{csv,md}` and `eval/results_rules.{csv,md}`. Sample results:

| metric | hybrid (LLM) | rules-only | target |
|---|---|---|---|
| extraction precision | 72% | **95%** ✓ | ≥ 90% |
| extraction recall | 64% | 64% | — |
| extraction F1 | 68% | **77%** | — |
| memory@1 accuracy | 62% | 62% | ≥ 85% |
| conflict resolution | 1/2 (50%) | 1/2 (50%) | — |
| avg latency / turn | 661 ms | **0 ms** | < 2000 ms ✓ |

**Honest interpretation:** rules-only hits the precision target because its predicates exactly match the curated ground truth. The LLM extracts more facts but with predicate variation that hurts strict matching. A predicate-canonicalisation layer (`extractor.py`) now maps synonyms (`is_studying`, `learning`, `majoring_in` → `studies`) onto one controlled vocabulary before they reach the graph; re-running the eval to quantify the gain is the natural next step.

### tests

```bash
python -m pytest tests/
```

51 tests across graph operations, conflict chains, reinforcement, polarity supersession, persistence, thread-safety, retrieval correctness, snapshot integrity, and edge cases.

---

## deploy

See [`DEPLOY.md`](DEPLOY.md) for a step-by-step.

**Recommended (free, no credit card):** Hugging Face Spaces with the Docker SDK.

```bash
docker build -t the-commonplace .
docker run -p 8000:7860 -e GROQ_API_KEY=$GROQ_API_KEY the-commonplace
```

---

## project layout

```
backend/                # FastAPI, graph, extractor, responder
  graph.py              time-aware property graph
  extractor.py          hybrid spaCy + Groq + rules, predicate canonicalisation
  query.py              token-overlap + 2-hop traversal
  responder.py          LLM reply grounded on retrieved facts
  session.py            per-visitor session isolation
  llm.py                Groq client
  main.py               app + WebSocket, session routing
  seed_data.py          demo seeding script
frontend/
  index.html            chat + live graph, single page
  styles.css            the interface theme
  app.js                D3 force-directed graph + chat + WS
eval/
  personachat_sample.json  curated dialogues with ground truth
  run_eval.py              precision / recall / memory / latency
tests/                  51 pytest cases
data/sessions/          persisted per-session graphs (auto-created)
Dockerfile              for HF Spaces / Render / any Docker host
```

---

## license

MIT — see [LICENSE](LICENSE).

This is a personal project, built in 2025 as an experiment in using time-aware
knowledge graphs as conversational memory. The work is released openly so the
architecture can be reused, extended, or argued with.

---

<div align="center">

made slowly, on paper first.

</div>
