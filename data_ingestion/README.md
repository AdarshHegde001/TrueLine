# data_ingestion: approved-knowledge ingestion + retrieval

Implements `RAG_INGESTION_CONTRACT.md`: versioned guideline / approved-fact sources in PostgreSQL,
BGE-M3 dense vectors (pgvector, `vector(1024)`), Postgres full-text search, brand-scoped hybrid
retrieval fused with Reciprocal Rank Fusion.

## Run it (5 steps)
```bash
cd data_ingestion
docker compose up -d db                                   # Postgres 16 + pgvector
pip install -r requirements.txt
export DATABASE_URL=postgresql://trueline:trueline@localhost:5432/trueline
python -m trueline_ingestion migrate
python -m trueline_ingestion seed                         # downloads BGE-M3 (~2.3 GB) first time
python -m trueline_ingestion search --brand brand_lumina --query "clinically proven to remove all wrinkles"
```
Add `--fake` to any command to skip the model download (test-only hash embedder; its tag says FAKE).

## Tests
```bash
export TEST_DATABASE_URL=postgresql://trueline:trueline@localhost:5432/trueline   # tests DROP knowledge_* tables!
pytest
```

## What the RAG workstream calls
```python
from trueline_ingestion import KnowledgeSearchRequest, search_knowledge
res = search_knowledge(KnowledgeSearchRequest(brand_id="brand_lumina", query="...", top_k=8))
```
Citation validation helpers (`trueline_ingestion/ingest.py`): `get_version(conn, brand_id, source_id, version_id)`
and `quote_exists(conn, brand_id, source_id, version_id, quote)`.

Errors: `ValidationFailed`, `NotFound`, `IndexingError`, `SearchServiceError` (`errors.py`).
Empty result = valid success. DB/embedding failure = `SearchServiceError`, never an empty list.

## Behaviour notes
- Searchable = chunk `ready` AND version `active` + `approved` AND it is the source's `current_version_id` AND chunk tag == current embedder tag.
- Update = new immutable version; old one becomes `inactive` (still resolvable by `get_version`).
- `deactivate_source` hides from retrieval; versions/chunks are retained (retention policy still open).
- Embedding failure => chunks `failed`, version stays `pending`; `reindex_stale` retries / rebuilds after a model change.
- Chunk offsets are Python `str` indices into `knowledge_source_versions.text` (`location.offset_base = "version_text"`).
- Keyword fallback only with `ALLOW_KEYWORD_FALLBACK=1`; responses are labelled `keyword`.
- Exact vector scan (no ANN index) until the dataset is large; HNSW statement is commented in the migration.

## Status (measured, not assumed)
- Tested: 17 pytest cases on PostgreSQL 16 + pgvector with the FAKE embedder (plumbing, scoping, versioning, failures).
- NOT yet tested: real BAAI/bge-m3 inference, retrieval quality, `EMBEDDING_REVISION` is still `main` (pin to a commit hash), HTTP/auth layer, docker compose.
