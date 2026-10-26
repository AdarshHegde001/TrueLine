"""CLI:  python -m trueline_ingestion {migrate|seed|search|reindex} [--fake]

--fake uses the test-only hash embedder (labelled FAKE in the DB). Never use it for the demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import db, ingest
from .config import get_settings
from .embeddings import BGEM3Embedder, FakeEmbedder
from .models import KnowledgeSearchRequest, SourceCreate
from .search import KnowledgeService

SEED = Path(__file__).resolve().parent.parent / "seed" / "demo_knowledge.json"


def main() -> None:
    p = argparse.ArgumentParser(prog="trueline_ingestion")
    p.add_argument("cmd", choices=["migrate", "seed", "search", "reindex"])
    p.add_argument("--brand", default="brand_lumina")
    p.add_argument("--query", default="")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()

    s = get_settings()
    conn = db.connect(s)
    if a.cmd == "migrate":
        print("applied:", db.migrate(conn))
        return
    emb = FakeEmbedder() if a.fake else BGEM3Embedder(s)
    if a.cmd == "seed":
        db.migrate(conn)
        for rec in json.loads(SEED.read_text()):
            out = ingest.create_source(conn, emb, SourceCreate(**rec), s)
            print(out.source_id, out.version_id, f"v{out.version}", out.title)
    elif a.cmd == "reindex":
        print("reindexed versions:", ingest.reindex_stale(conn, emb, s))
    else:
        svc = KnowledgeService(emb, s)
        res = svc.search(KnowledgeSearchRequest(brand_id=a.brand, query=a.query, top_k=a.top_k))
        print(f"method={res.retrieval_method}  n={len(res.results)}")
        for r in res.results:
            print(f"  #{r.retrieval.rank} [{r.source_type}] {r.title} ({r.source_id}/{r.version_id} v{r.version})"
                  f"\n      {r.chunk_text[:110]}")


if __name__ == "__main__":
    main()
