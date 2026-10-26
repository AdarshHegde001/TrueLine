from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from .config import Settings, get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def connect(settings: Settings | None = None) -> psycopg.Connection:
    s = settings or get_settings()
    return psycopg.connect(s.database_url, row_factory=dict_row)


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply every migrations/*.sql in order (all are idempotent)."""
    applied = []
    for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
        conn.execute(f.read_text())
        applied.append(f.name)
    conn.commit()
    return applied


def vec_literal(v: list[float]) -> str:
    """pgvector text literal; used with %s::vector so no extra driver package is needed."""
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"
