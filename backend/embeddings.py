from __future__ import annotations

import os
import threading
from typing import Optional


# Récupération sémantique — optionnelle.
#
# Si une librairie d'embeddings est installée, on encode requêtes et faits en
# vecteurs pour comparer par le *sens* (et pas seulement les mots partagés).
# Backends tentés dans l'ordre : fastembed (ONNX, léger, sans PyTorch) puis
# sentence-transformers. Si rien n'est dispo, ``available`` reste False et la
# couche de récupération se rabat sur le lexical — même philosophie de
# dégradation gracieuse que le reste du projet.

# bge-* recommande de préfixer la REQUÊTE (pas les documents) par cette
# instruction pour la recherche asymétrique requête→passage.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384 dim, quantifié, CPU
_ST_MODEL = "all-MiniLM-L6-v2"


class _Embedder:
    """Encodeur paresseux, thread-safe, avec cache des vecteurs de faits."""

    def __init__(self):
        self._model = None
        self._backend: Optional[str] = None
        self._np = None
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._tried = False

    def _lazy_init(self) -> None:
        if self._tried:
            return
        self._tried = True
        # Coupe-circuit : on peut désactiver la sémantique (déploiement léger,
        # tests, machine sans la lib) → retour au lexical pur.
        if os.getenv("KG_DISABLE_EMBEDDINGS", "").strip().lower() in {"1", "true", "yes"}:
            return
        try:
            import numpy as np
            self._np = np
        except Exception:
            return  # sans numpy, pas de calcul vectoriel possible

        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(_FASTEMBED_MODEL)
            self._backend = "fastembed"
            return
        except Exception:
            pass

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_ST_MODEL)
            self._backend = "sentence-transformers"
            return
        except Exception:
            pass

    @property
    def available(self) -> bool:
        self._lazy_init()
        return self._model is not None

    @property
    def backend(self) -> Optional[str]:
        self._lazy_init()
        return self._backend

    # ── encodage ────────────────────────────────────────────────────────

    def _embed_raw(self, texts: list[str]) -> list:
        np = self._np
        if self._backend == "fastembed":
            return [np.asarray(v, dtype="float32") for v in self._model.embed(texts)]
        # sentence-transformers
        arr = self._model.encode(texts, normalize_embeddings=False)
        return [np.asarray(v, dtype="float32") for v in arr]

    def _normalize(self, v):
        np = self._np
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_docs(self, texts: list[str]) -> Optional[list]:
        """Encode des textes de faits, avec cache (un fait répété n'est pas
        ré-encodé à chaque requête)."""
        if not self.available:
            return None
        missing = [t for t in texts if t not in self._cache]
        if missing:
            with self._lock:
                todo = [t for t in missing if t not in self._cache]
                if todo:
                    for t, v in zip(todo, self._embed_raw(todo)):
                        self._cache[t] = self._normalize(v)
        return [self._cache[t] for t in texts]

    def encode_query(self, text: str):
        if not self.available:
            return None
        q = (_BGE_QUERY_PREFIX + text) if self._backend == "fastembed" else text
        return self._normalize(self._embed_raw([q])[0])

    def cosine(self, a, b) -> float:
        # vecteurs déjà normalisés → le produit scalaire EST le cosinus
        return float(a @ b)


# Singleton partagé par toute l'application.
EMBEDDER = _Embedder()
