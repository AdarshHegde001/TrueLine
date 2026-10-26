"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_model_name: str      # HF model id
    embedding_revision: str        # PIN THIS to a commit hash in deployment
    embedding_dim: int = 1024
    chunking_version: str = "chunk-v1"
    chunk_max_chars: int = 800
    candidate_pool: int = 30       # per-method candidates before fusion
    allow_keyword_fallback: bool = False  # dev only; response is labelled 'keyword'

    @property
    def embedding_tag(self) -> str:
        """Stored in knowledge_chunks.embedding_model; queries must match it."""
        return f"{self.embedding_model_name}@{self.embedding_revision}|dense|normalized"


def get_settings() -> Settings:
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL", "postgresql://trueline:trueline@localhost:5432/trueline"
        ),
        embedding_model_name=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_revision=os.environ.get("EMBEDDING_REVISION", "main"),
        allow_keyword_fallback=os.environ.get("ALLOW_KEYWORD_FALLBACK", "0") == "1",
    )
