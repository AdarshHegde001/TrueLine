"""Embedding adapters for production Gemini and optional local BGE-M3."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class GeminiEmbedding2Embedder:
    """Remote Gemini Embedding 2 adapter; uses the existing 1024-d pgvector column."""

    dim = 1024
    batch_size = 16
    max_attempts = 4

    def __init__(self, settings: Settings | None = None):
        self.model = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2").strip()
        self.api_key = os.environ.get("GEMINI_EMBEDDING_API_KEY", "").strip()
        self.timeout = float(os.environ.get("GEMINI_EMBEDDING_TIMEOUT_SECONDS", "60"))
        self.endpoint = os.environ.get(
            "GEMINI_EMBEDDING_ENDPOINT",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        self.dim = (settings or get_settings()).embedding_dim
        if not self.api_key:
            raise RuntimeError("GEMINI_EMBEDDING_API_KEY is required when INGESTION_EMBEDDER=gemini")
        if not self.model or self.dim != 1024:
            raise RuntimeError("Gemini embedding configuration must use a model and the current 1024-d schema")
        self.tag = f"{self.model}@stable|dense|normalized|dim-{self.dim}|gemini-search-prompt-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Gemini Embedding 2 recommends distinct document/query task formatting.
        formatted = [f"title: none | text: {text}" for text in texts]
        return self._embed_formatted(formatted)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_formatted([f"task: search result | query: {text}"])[0]

    def _embed_formatted(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        model_path = quote(self.model.removeprefix("models/"), safe="-")
        url = f"{self.endpoint}/models/{model_path}:batchEmbedContents"
        for offset in range(0, len(texts), self.batch_size):
            batch = texts[offset:offset + self.batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.model.removeprefix('models/')}",
                        "content": {"parts": [{"text": text}]},
                        "embedContentConfig": {"outputDimensionality": self.dim},
                    }
                    for text in batch
                ]
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                method="POST",
            )
            response_data = self._post_with_retries(request)
            embeddings = response_data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise RuntimeError("Gemini embedding response count did not match the request")
            for embedding in embeddings:
                values = embedding.get("values") if isinstance(embedding, dict) else None
                if not isinstance(values, list) or len(values) != self.dim:
                    raise RuntimeError(f"Gemini embedding response must contain {self.dim} values")
                vector = [float(value) for value in values]
                if not all(math.isfinite(value) for value in vector):
                    raise RuntimeError("Gemini embedding response contained a non-finite value")
                vectors.append(vector)
        return vectors

    def _post_with_retries(self, request: Request) -> dict:
        for attempt in range(self.max_attempts):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code <= 599
                if not retryable or attempt + 1 == self.max_attempts:
                    raise RuntimeError(f"Gemini embedding API returned HTTP {exc.code}") from exc
            except URLError as exc:
                if attempt + 1 == self.max_attempts:
                    raise RuntimeError("Gemini embedding API request failed") from exc
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Gemini embedding API request failed")


class FakeEmbedder:
    """TEST ONLY. Hashed bag-of-words into 1024 dims. Not semantic, not BGE-M3.
    Its tag says so, so fake vectors can never be confused with real ones."""

    dim = 1024
    tag = "FAKE-hash-bow-test-only|document-type-prefix-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def create_embedder(mode: str | None = None, settings: Settings | None = None) -> Embedder:
    """Create the configured embedding provider; production defaults to remote Gemini."""
    selected = (mode or os.environ.get("INGESTION_EMBEDDER", "gemini")).strip().lower()
    if selected == "gemini":
        return GeminiEmbedding2Embedder(settings)
    if selected in {"bge", "bge-m3"}:
        return BGEM3Embedder(settings)
    if selected == "fake":
        return FakeEmbedder()
    raise ValueError(f"Unsupported INGESTION_EMBEDDER: {selected}")
