# data_ingestion: approved-knowledge ingestion + retrieval

Implements `RAG_INGESTION_CONTRACT.md`: versioned guideline / approved-fact sources in PostgreSQL,
BGE-M3 dense vectors (pgvector, `vector(1024)`), Postgres full-text search, brand-scoped hybrid
retrieval fused with Reciprocal Rank Fusion.

## Run it (5 steps)
```bash
cd data_ingestion
docker compose up -d db                                   # Postgres 16 + pgvector
pip install -r requirements.txt
export DATABASE_URL=postgresql://trueline:trueline@localhost:5433/trueline
python -m trueline_ingestion migrate
python -m trueline_ingestion seed                         # downloads BGE-M3 (~2.3 GB) first time
python -m trueline_ingestion search --brand brand_lumina --query "clinically proven to remove all wrinkles"
```
Add `--fake` to any command to skip the model download (test-only hash embedder; its tag says FAKE).

## Tests
```bash
docker compose exec db createdb -U trueline trueline_test
export TEST_DATABASE_URL=postgresql://trueline:trueline@localhost:5433/trueline_test # test fixture DROPs knowledge_* tables
pytest
```

## What the RAG workstream calls
```python
from trueline_ingestion import KnowledgeSearchRequest, get_deterministic_rules, search_knowledge
res = search_knowledge(KnowledgeSearchRequest(brand_id="brand_lumina", query="...", top_k=8))
rules = get_deterministic_rules(brand_id="brand_lumina", channel="advertisement")
```
Search returns a mapping with the canonical `rag.retrieval.models.KnowledgeSearchResponse` fields;
the RAG adapter validates that mapping at the boundary.
Each uploaded source also carries an organization-selected `document_type`:
`brand_guidelines`, `tone_of_voice`, `approved_claims`, `product_facts`,
`glossary`, `asset_rules`, `legal_compliance`, or `other_reference`. It is stored on the immutable version and every
chunk, prefixed into BGE-M3 embedding input, and returned in each retrieval result. Set
`document_types=("tone_of_voice",)` on `KnowledgeSearchRequest` to scope retrieval to a
specialist agent's evidence; an empty tuple searches every type.
The embedding-input format is versioned in the stored embedding tag. After upgrading an existing
database, reindex approved sources with the same embedder configured for the backend (`python -m
trueline_ingestion reindex`; add `--fake` only when the database was indexed with the fake embedder).
Citation validation helpers (`trueline_ingestion/ingest.py`): `get_version(conn, brand_id, source_id, version_id)`
and `quote_exists(conn, brand_id, source_id, version_id, quote)`. Structured rules are saved on immutable
versions as `rule_data` JSONB and exposed by `get_deterministic_rules`; rules only become available when
their containing version is active and approved.

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
