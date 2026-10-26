"""Embedders. Production = BGE-M3 dense (1024-d, L2-normalised).

Documents and queries MUST use the same model/revision (settings.embedding_tag)."""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import Settings, get_settings


class Embedder(Protocol):
    tag: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class BGEM3Embedder:
    """Dense BAAI/bge-m3 via sentence-transformers. Heavy import is lazy."""

    def __init__(self, settings: Settings | None = None):
        s = settings or get_settings()
        from sentence_transformers import SentenceTransformer  # lazy

        kwargs = {} if s.embedding_revision == "main" else {"revision": s.embedding_revision}
        self._model = SentenceTransformer(s.embedding_model_name, **kwargs)
        self.tag = s.embedding_tag
        self.dim = s.embedding_dim
        got = self._model.get_sentence_embedding_dimension()
        if got != self.dim:
            raise RuntimeError(f"model dim {got} != expected {self.dim}; refusing to run")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=16)
        return [v.tolist() for v in vecs]


class FakeEmbedder:
    """TEST ONLY. Hashed bag-of-words into 1024 dims. Not semantic, not BGE-M3.
    Its tag says so, so fake vectors can never be confused with real ones."""

    dim = 1024
    tag = "FAKE-hash-bow-test-only"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out
