# Data Ingestion ↔ RAG Contract

**Status: proposed; confirm with both workstreams before treating it as a fixed API.**

This document defines the handoff between the data ingestion workstream and the RAG/review workstream. The ingestion service owns approved brand knowledge and prepares its search index. The RAG service owns issuing review-time searches, using retrieved knowledge, and producing evidence-backed findings. The frontend should not access the index directly.

The agreed knowledge store is PostgreSQL with the `pgvector` extension for dense embeddings, alongside PostgreSQL full-text search for lexical matching. The agreed embedding model is `BAAI/bge-m3`; its dense output dimension is 1024, so the chunk embedding column is `vector(1024)`. This contract fixes the storage and retrieval boundary the RAG workstream can build against. The initial goal is a small, reliable text-document flow.

## Responsibilities

### Ingestion workstream

Implement the path that lets an authorized brand administrator add and manage approved source material, and makes that material available to retrieval. In particular:

1. Accept guideline/rule and approved-fact records associated with a brand, with an organization-selected `document_type` describing the document's review purpose.
2. Validate required fields and reject malformed or unsupported records with useful errors.
3. Assign stable source IDs and immutable version IDs (or an equivalent monotonically increasing version) to each published source revision.
4. Preserve the source text and its provenance. Do not silently rewrite the authoritative text during chunking or embedding.
5. Split long sources into retrievable chunks while retaining their parent source ID, version, and location in the original source. Store each chunk's exact text; chunking must not replace or mutate the authoritative source text.
6. Create one dense embedding for each approved chunk using `BAAI/bge-m3` and store it in PostgreSQL through `pgvector` as `vector(1024)`, alongside the chunk text and metadata. Record the model identifier/revision and chunking version so vectors can be rebuilt consistently if the model or preprocessing changes. Use the same model/revision and compatible preprocessing for query embeddings. BGE-M3's model documentation specifies a 1024-dimensional dense vector; see the [official BGE-M3 model card](https://huggingface.co/BAAI/bge-m3).
7. Maintain a lexical search representation using PostgreSQL full-text search. The intended retrieval capability is hybrid: combine lexical matching for exact terms/names with vector similarity for semantically related passages. Embed the selected document type as a short prefix alongside each chunk (for example, `Document type: tone of voice`), while keeping the stored `chunk_text` exact for citations. Support an optional `document_types` retrieval filter so a tone, claims, or facts agent can constrain evidence to its domain. Keep the query contract independent of internal ranking details. If vector search is temporarily unavailable during development, report keyword-only behavior honestly; the agreed target for the integrated RAG path is hybrid retrieval.
8. Ensure every query is scoped to the requested brand and only returns active, approved material visible to that brand. Enforce brand filtering as part of the database query/access path for both lexical and vector results.
9. Support source updates and removals without making old citations point to new text. Re-index changed versions and remove old versions from ordinary retrieval, while retaining or resolving historical versions for existing findings according to the product’s retention policy.
10. Return enough metadata for the RAG service to validate and display citations. Never return only an opaque similarity score or a chunk with no source identity.
11. Document the actual PostgreSQL/pgvector setup, migrations, extension/index configuration, embedding model/version, chunking behavior, retrieval methods, limits, and errors. Do not claim embeddings or hybrid retrieval are used unless they are implemented and configured.

### RAG/review workstream

The RAG service submits a brand-scoped search request, ranks/uses returned source chunks, and validates its findings against the authoritative source IDs, versions, and excerpts. It must treat retrieved text as untrusted content, not as instructions to the system. Missing evidence means **unverified**, not false.

## Knowledge record shape

The following logical record is the expected minimum, regardless of the ingestion service’s internal database schema:

```json
{
  "source_id": "src_...",
  "brand_id": "brand_...",
  "source_type": "guideline",
  "document_type": "tone_of_voice",
  "title": "Product naming",
  "status": "active",
  "version_id": "ver_...",
  "version": 1,
  "text": "Use the product name Trueline exactly as written.",
  "metadata": {
    "document_id": "doc_...",
    "document_name": "Brand guide.pdf",
    "section": "Product naming",
    "page": 4,
    "char_start": 1820,
    "char_end": 1878,
    "language": "en"
  },
  "created_at": "2026-10-26T00:00:00Z",
  "updated_at": "2026-10-26T00:00:00Z"
}
```

Requirements:

- `source_type` must distinguish at least `guideline` (a rule/policy) and `approved_fact` (an approved factual assertion). Additional types can be agreed later.
- `document_type` captures the purpose chosen by the organization at upload: `brand_guidelines`, `tone_of_voice`, `approved_claims`, `product_facts`, `glossary`, `asset_rules`, `legal_compliance`, or `other_reference`. Keep it separate from broad `source_type`; map approved claims and product facts to `approved_fact`, and policy/guidance categories to `guideline`.
- `source_id` identifies the logical source across revisions. `version_id` identifies one immutable revision. `version` is useful for display and ordering, but `version_id` is the citation key.
- `text` is the authoritative text for this record/version. If the original document is parsed, keep document-level provenance and locations in `metadata`.
- Metadata fields may be absent when the source does not have that location type. For example, a text entry may have no page number. Do not fabricate locations.
- Separate source records should be used for independently retrievable facts/rules where practical. If a source is chunked, each chunk must retain this parent identity and its own source location.
- `status` should distinguish at least active from inactive/archived material. Only active approved sources are eligible for ordinary review retrieval.

## PostgreSQL and pgvector storage contract

Use the following logical tables and column meanings as the shared schema. The backend may add columns, but must preserve these names/semantics or provide a compatibility view so the RAG retrieval module does not need to be rewritten. PostgreSQL owns canonical source text and metadata; pgvector stores/searches the dense chunk vectors.

### `knowledge_sources`

One row per logical source across versions:

| Column | Type / constraint | Meaning |
|---|---|---|
| `source_id` | UUID or stable text ID, primary key | Stable identity used by citations across revisions |
| `brand_id` | UUID or stable text ID, not null, indexed | Owning brand; every retrieval query filters by this |
| `source_type` | text, not null | At minimum `guideline` or `approved_fact` |
| `title` | text, not null | Human-readable source/rule title |
| `current_version_id` | UUID or stable text ID, nullable FK | Current approved version, if one exists |
| `created_at`, `updated_at` | timestamptz, not null | Audit timestamps |

### `knowledge_source_versions`

One immutable row per source revision:

| Column | Type / constraint | Meaning |
|---|---|---|
| `version_id` | UUID or stable text ID, primary key | Immutable revision ID used to pin citations |
| `source_id` | FK to `knowledge_sources`, not null | Logical source identity |
| `version_number` | integer, not null; unique with `source_id` | Ordered revision number |
| `brand_id` | same ID type, not null, indexed | Denormalized for safe scoped reads; must match source brand |
| `status` | text, not null | `pending`, `active`, `inactive`, or `archived` |
| `approval_status` | text, not null | At minimum distinguish `approved` from `pending`/`rejected` |
| `text` | text, not null | Canonical authoritative text for this revision |
| `rule_data` | JSONB, not null, default `[]` | Optional structured deterministic rules explicitly confirmed/entered by an admin; never inferred by code from prose alone |
| `document_type` | text, not null | Purpose selected at upload; copied to every chunk for retrieval filtering |
| `document_id` | nullable stable ID | Uploaded document identity, when applicable |
| `document_name` | nullable text | Original filename/title, when available |
| `metadata` | JSONB, not null, default `{}` | Section, page, language, and other provenance; no invented values |
| `created_at` | timestamptz, not null | Revision creation time |

`rule_data` is an array. Each entry has a `rule_id` and `kind`, and the retrieval/rule adapter maps it to the RAG typed rule models. Minimum supported forms:

```json
[
  {
    "rule_id": "rule_123",
    "kind": "terminology",
    "mode": "replace_with_preferred",
    "term": "Tru Line",
    "replacement": "Trueline",
    "case_sensitive": false,
    "whole_word": true,
    "severity": "medium",
    "applies_to_channels": []
  },
  {
    "rule_id": "rule_124",
    "kind": "required_text",
    "phrase": "Terms and conditions apply.",
    "case_sensitive": false,
    "applies_to_channels": ["advertisement"]
  }
]
```

The source identity/version and source excerpt for each structured rule come from the containing approved version; the adapter attaches them as the rule's `SourceReference`. Supported terminology modes are `banned` and `replace_with_preferred`. Required text is checked as a literal phrase with whitespace-insensitive matching. Admin confirmation is required before rule data is active. Arbitrary natural-language rules remain available to retrieval/AI review and must not be treated as deterministic checks unless they have an explicit supported structured representation.

### `knowledge_chunks`

One row per retrievable passage. A unique `(version_id, chunk_index)` constraint provides stable ordering within a version:

| Column | Type / constraint | Meaning |
|---|---|---|
| `chunk_id` | UUID or stable text ID, primary key | Stable chunk identity |
| `brand_id` | same ID type, not null, indexed | Must match the parent source/version brand |
| `source_id` | FK to `knowledge_sources`, not null | Citation source |
| `version_id` | FK to `knowledge_source_versions`, not null | Exact immutable source revision |
| `chunk_index` | integer, not null | Order of this chunk within the version |
| `chunk_text` | text, not null | Exact source passage returned to RAG and used for citations |
| `document_type` | text, not null | Copied from the source version; included as context in the embedding input and filterable in SQL |
| `location` | JSONB, not null, default `{}` | Section/page/character offsets in the canonical source representation, when known |
| `search_tsv` | `tsvector`, maintained from `chunk_text` | PostgreSQL full-text search representation |
| `embedding` | `vector(1024)`, not null when indexed | Dense BGE-M3 embedding for `Document type: <human-readable type>\n<chunk_text>` |
| `embedding_model` | text, not null | Must identify `BAAI/bge-m3`, revision/configuration, and embedding-input format version |
| `chunking_version` | text, not null | Version of the chunking/preprocessing procedure |
| `index_status` | text, not null | At minimum `pending`, `ready`, or `failed`; only `ready` rows are searchable |
| `created_at`, `updated_at` | timestamptz, not null | Index lifecycle timestamps |

Use foreign keys and, where supported by the chosen ID layout, composite constraints to prevent a chunk's `brand_id` from disagreeing with its parent source/version. Keep a B-tree index for brand/status filtering and a GIN index for `search_tsv`. Create a pgvector cosine-distance HNSW index on `embedding` for vector candidate search, plus any needed brand/status filtering indexes. Keep the distance operator, index opclass, and query ordering consistent (cosine distance / `vector_cosine_ops`). The embedding model dimension is fixed here at 1024; an embedding from another model/dimension must not be inserted into this column.

Enable the PostgreSQL `vector` extension through a reproducible migration before creating the vector column or index. PostgreSQL full-text search supplies lexical matching; pgvector supplies dense semantic matching. The retrieval implementation combines candidates from both into one bounded result list. Keep both candidate queries brand-scoped and restricted to approved, active source versions and ready chunks.

For the first working integration, PostgreSQL may perform exact vector comparison before an approximate-nearest-neighbor index is justified by dataset size or measured latency. If an approximate index such as HNSW or IVFFlat is introduced, document its chosen parameters and evaluate retrieval quality against the project dataset. The index is a performance aid; the source text and metadata remain the citation authority. Refer to the [pgvector project documentation](https://github.com/pgvector/pgvector) for supported PostgreSQL extension types, distance operators, and index options, and the [PostgreSQL full-text search documentation](https://www.postgresql.org/docs/current/textsearch.html) for lexical search.

Embedding/index lifecycle requirements:

1. Only approved, active source versions are eligible for ordinary retrieval and embedding generation.
2. Chunk text, source identity, version, and location are stored together so a vector match can always be traced back to its authoritative passage.
3. When source text, chunking logic, embedding-input format (including the document-type prefix), or embedding model changes, regenerate affected chunks/vectors and avoid mixing incompatible vector dimensions or model versions in the same search operation. The implementation tag uses `document-type-prefix-v1`; reindex existing approved sources after upgrading to this format.
4. Model unavailability or an embedding/indexing error must be visible as an ingestion/indexing failure or pending state. Do not mark a source searchable when it has not been indexed successfully.
5. Keep PostgreSQL migrations/configuration reproducible. The `vector` extension must be enabled in the target database, and extension/index setup must be documented for local development and deployment.

Use the official `BAAI/bge-m3` dense embedding output. The model supports dense, sparse, and multi-vector modes; this contract uses its dense 1024-dimensional output with pgvector plus PostgreSQL full-text search. Do not silently substitute the model or mix vector modes. Record the exact model revision and inference configuration in configuration/docs. Do not add a separate vector database for this flow without a demonstrated need and explicit team agreement.

## Stable RAG retrieval interface

The ingestion/retrieval module must expose this logical Python operation to the review pipeline. The canonical Pydantic request/response models live in `rag.retrieval.models` (`KnowledgeSearchRequest` and `KnowledgeSearchResponse`); ingestion should import and return those models, or an exactly compatible mapping. If the backend exposes retrieval over HTTP, keep these fields identical and implement an adapter. RAG business logic depends on this shape, not on SQL table details.

```python
def search_knowledge(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """Return approved, active brand knowledge relevant to a focused query."""
```

`KnowledgeSearchRequest` fields:

| Field | Type | Requirement |
|---|---|---|
| `brand_id` | string | Required; enforced in every SQL candidate query |
| `query` | string | Required focused text query; embed with the same `BAAI/bge-m3` revision used for documents |
| `source_types` | list of strings | Optional; defaults to guideline + approved_fact |
| `document_types` | list of strings | Optional; filters to the selected purpose values. Empty means all document types. |
| `top_k` | integer | Optional; default 8, valid range 1–20 |

`KnowledgeSearchResponse` contains `brand_id`, `results`, and `retrieval_method` (`hybrid`, `vector`, or `keyword`). Each result contains `source_id`, `version_id`, `brand_id`, `source_type`, `document_type`, `title`, `version_number`, `chunk_id`, exact `chunk_text`, authoritative `source_text` (or a reliable source/version lookup), `location`, `embedding_model`, and rank/score metadata. Scores are retrieval signals, not confidence or proof of support. No match is a successful response with an empty `results` list. Database/index failures must raise/return an explicit service error, not masquerade as no matches.

The interface must never return unapproved/inactive material. A brand authorization layer must validate the caller's access before invoking the function; passing a brand ID alone is not authorization. For the initial local integration, callers may provide an already-authorized brand context from the backend, but the function must still apply the brand filter in SQL.

## Retrieval interface

The exact transport (direct Python service call or HTTP endpoint) should match the backend architecture the team agrees on. The logical request and response below are the contract either way. Do not build a second public API solely to satisfy this draft if the backend already provides an agreed internal interface.

In addition to semantic/lexical search, expose a structured-rule read operation for deterministic checks. Search is query-dependent and may not return every terminology/disclaimer rule, so the RAG service must not rely on top-k search to load rules that need to run against the full submission.

```python
def get_deterministic_rules(*, brand_id: str, channel: str | None) -> list[dict]:
    """Return all active, approved structured rules applicable to this brand/channel."""
```

Each returned object is one `rule_data` entry augmented with its containing approved source identity/version:

```json
{
  "rule_id": "rule_123",
  "kind": "terminology",
  "mode": "replace_with_preferred",
  "term": "Tru Line",
  "replacement": "Trueline",
  "case_sensitive": false,
  "whole_word": true,
  "severity": "medium",
  "applies_to_channels": [],
  "source_id": "src_...",
  "version_id": "ver_...",
  "source_type": "guideline",
  "source_title": "Product naming",
  "source_excerpt": "Use the product name Trueline exactly as written. Do not use Tru Line."
}
```

For `required_text`, return `kind`, `rule_id`, `phrase`, `case_sensitive`, `applies_to_channels`, and the same source fields. The operation must return all active approved applicable structured rules, not a ranked top-k subset; an empty list means no structured deterministic rules are configured. Only the ingestion/admin workflow marks structured rule data approved. The RAG adapter at `rag.rules.rules_from_ingestion_records` rejects malformed/unknown rule kinds rather than silently skipping them. If no channel is supplied, channel-specific records are not applicable and must not be represented as passed.

### Request

```json
{
  "brand_id": "brand_...",
  "query": "What product name is allowed in this sentence?",
  "source_types": ["guideline", "approved_fact"],
  "document_types": ["tone_of_voice"],
  "top_k": 8
}
```

`brand_id` and `query` are required. `source_types` and `top_k` are optional; establish a documented default and maximum for `top_k`. The caller may issue multiple focused queries for separate passages or claims. Do not require the ingestion service to interpret a whole submitted document as one search query.

### Response

```json
{
  "brand_id": "brand_...",
  "retrieval_method": "hybrid",
  "results": [
    {
      "source_id": "src_...",
      "brand_id": "brand_...",
      "source_type": "guideline",
      "document_type": "tone_of_voice",
      "title": "Product naming",
      "version_id": "ver_...",
      "version_number": 1,
      "text": "Use the product name Trueline exactly as written.",
      "source_text": "Use the product name Trueline exactly as written.",
      "chunk_id": "chunk_...",
      "chunk_text": "Use the product name Trueline exactly as written.",
      "embedding_model": "BAAI/bge-m3@<revision>",
      "rank": 1,
      "score": 0.92,
      "location": {
        "document_id": "doc_...",
        "document_name": "Brand guide.pdf",
        "section": "Product naming",
        "page": 4,
        "char_start": 1820,
        "char_end": 1878
      }
    }
  ]
}
```

Requirements:

- Every result must include `brand_id`, `source_id`, `source_type`, `document_type`, `version_id`, `version_number`, and the exact retrieved `chunk_text`. Include `source_text` when practical; otherwise provide an authoritative lookup by `source_id` + `version_id` for citation validation.
- `location` must refer to the original material and identify the chunk’s location where known. Use character offsets only when they are measured against a clearly defined canonical text representation.
- Response-level `retrieval_method` identifies the method actually used, such as `keyword`, `vector`, or `hybrid`. A result's `score` is optional and must not be presented as confidence that a source supports a claim.
- Results must be deterministic in shape, even when the result list is empty. An empty result is a valid response, not an error.
- Results must be scoped to the requested brand on the server/data-access layer, not merely filtered by the caller after retrieval.
- Retrieval relevance is not semantic proof. The RAG service remains responsible for checking whether a source actually supports a finding.

## Errors and behavior

Use the host service’s established error format if one exists. At minimum distinguish:

- Invalid request (missing brand/query, invalid `top_k`, unsupported type): validation error.
- Unknown or inaccessible brand: not found/forbidden according to the authorization design; do not reveal another brand’s existence or contents.
- Retrieval/index unavailable: explicit service error. Do not return a fabricated empty success result on an infrastructure failure.
- No matching sources: successful response with `results: []`.
- Partial indexing or stale index: report a visible status/error or make the source unavailable; do not cite it as current and approved without resolving the state.

## Citation and version guarantees

1. A result’s `source_id` and `version_id` must resolve to the exact approved record revision returned by retrieval.
2. Updating a source creates a new immutable revision; it must not change the text behind an existing `version_id`.
3. The RAG service must be able to verify that a returned quote is an exact substring of the stored source/chunk text for that version.
4. Deactivation/removal must prevent the source from appearing in new ordinary retrieval results. Historical findings need a product-level retention decision; do not mutate their cited revision in place.
5. A source reference proves identity and provenance only. It does not by itself prove that the source entails or supports a proposed finding.

## Initial implementation boundary

For the first integration, support:

- Text-based guideline and approved-fact records.
- One brand-scoped query operation.
- PostgreSQL storage for versioned source/chunk records and provenance.
- A bounded retrieval result count using PostgreSQL full-text search and pgvector vector search (hybrid retrieval) with `BAAI/bge-m3` dense 1024-dimensional embeddings. Keyword-only search is allowed only as a temporary development fallback and must be labeled accurately in responses.
- Stable source/version IDs and source locations where available.
- Explicit empty-result and failure behavior.
- A small fictional/demo dataset for local development. Do not commit real confidential brand material.

Keep sophisticated ranking, OCR/scanned documents, and ingestion UI details out of the initial handoff unless the team has already approved them. PostgreSQL + pgvector is the planned vector-storage path; the logical result contract should allow ranking details to evolve without changing how RAG validates citations.

## Acceptance checklist for the ingestion handoff

- [ ] Ingestion can create or load guideline and approved-fact records with stable brand/source/version identity.
- [ ] A source update produces a new immutable version.
- [ ] PostgreSQL stores versioned source/chunk records and their provenance; the target database has a reproducible `pgvector` extension setup.
- [ ] BGE-M3 dense vectors (1024 dimensions) are generated with a recorded model revision/configuration and stored per chunk.
- [ ] A query accepts a brand ID, focused text query, optional source types, and bounded `top_k`.
- [ ] A structured-rule operation returns all active, approved deterministic rules applicable to a brand/channel, with source ID/version/excerpt attached.
- [ ] Retrieval supports the agreed lexical/vector strategy and reports the actual method used; keyword-only fallback is labeled honestly.
- [ ] Every returned chunk includes the required identity, exact text, and available provenance/location.
- [ ] A query for one brand cannot return another brand’s sources.
- [ ] Inactive/unapproved sources are excluded from ordinary results.
- [ ] No-match returns an empty result list; service/index failure is distinguishable from no match.
- [ ] The RAG workstream can resolve and validate a citation against the same source version.
- [ ] Contract examples and any implementation-specific endpoint/function name are recorded in the project docs once agreed.

## Decisions still to confirm together

- Is retrieval called as an internal Python interface or through an HTTP endpoint?
- Which component enforces the caller’s brand authorization, and what identity/context is passed?
- What is the canonical representation used for source text and character offsets?
- How long must superseded/deactivated source versions remain available for historical findings?
- Which structured fields distinguish approved facts from rules in the actual ingestion model?
- What exact BGE-M3 model revision and inference runtime/configuration will be used in deployment, and how will that configuration be pinned?
- What retention period applies to superseded/deactivated source versions for historical findings?

The embedding model (`BAAI/bge-m3`), vector store (`PostgreSQL` + `pgvector`), and logical retrieval fields are agreed in this contract. Resolve the remaining deployment/runtime, authorization, and retention decisions with the backend and ingestion owners before treating this as an implemented API. Update this contract when code is implemented; keep it aligned with the code and API documentation.
