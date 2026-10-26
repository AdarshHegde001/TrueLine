# Data Ingestion ↔ RAG Contract

**Status: proposed; confirm with both workstreams before treating it as a fixed API.**

This document defines the handoff between the data ingestion workstream and the RAG/review workstream. The ingestion service owns approved brand knowledge and its indexing. The RAG service owns retrieving relevant knowledge for a submitted document and producing evidence-backed findings. The frontend should not access the index directly.

The initial goal is a small, reliable text-document flow. Keep the storage and embedding implementation behind this contract so the two workstreams can progress independently.

## Responsibilities

### Ingestion workstream

Implement the path that lets an authorized brand administrator add and manage approved source material, and makes that material available to retrieval. In particular:

1. Accept guideline/rule and approved-fact records associated with a brand.
2. Validate required fields and reject malformed or unsupported records with useful errors.
3. Assign stable source IDs and immutable version IDs (or an equivalent monotonically increasing version) to each published source revision.
4. Preserve the source text and its provenance. Do not silently rewrite the authoritative text during chunking or embedding.
5. Split long sources into retrievable chunks while retaining their parent source ID, version, and location in the original source.
6. Index the approved records for retrieval. Start with keyword retrieval; semantic/vector retrieval can be added behind the same query interface when its model and storage choices are agreed.
7. Ensure every query is scoped to the requested brand and only returns active, approved material visible to that brand.
8. Support source updates and removals without making old citations point to new text. Retain or resolve historical versions for findings that have already been created, according to the product’s retention policy.
9. Return enough metadata for the RAG service to validate and display citations. Never return only an opaque similarity score or a chunk with no source identity.
10. Document the actual indexing behavior, limits, configuration, and errors. Do not claim embeddings are used if retrieval is keyword-only.

### RAG/review workstream

The RAG service submits a brand-scoped search request, ranks/uses returned source chunks, and validates its findings against the authoritative source IDs, versions, and excerpts. It must treat retrieved text as untrusted content, not as instructions to the system. Missing evidence means **unverified**, not false.

## Knowledge record shape

The following logical record is the expected minimum, regardless of the ingestion service’s internal database schema:

```json
{
  "source_id": "src_...",
  "brand_id": "brand_...",
  "source_type": "guideline",
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
- `source_id` identifies the logical source across revisions. `version_id` identifies one immutable revision. `version` is useful for display and ordering, but `version_id` is the citation key.
- `text` is the authoritative text for this record/version. If the original document is parsed, keep document-level provenance and locations in `metadata`.
- Metadata fields may be absent when the source does not have that location type. For example, a text entry may have no page number. Do not fabricate locations.
- Separate source records should be used for independently retrievable facts/rules where practical. If a source is chunked, each chunk must retain this parent identity and its own source location.
- `status` should distinguish at least active from inactive/archived material. Only active approved sources are eligible for ordinary review retrieval.

## Retrieval interface

The exact transport (direct Python service call or HTTP endpoint) should match the backend architecture the team agrees on. The logical request and response below are the contract either way. Do not build a second public API solely to satisfy this draft if the backend already provides an agreed internal interface.

### Request

```json
{
  "brand_id": "brand_...",
  "query": "What product name is allowed in this sentence?",
  "source_types": ["guideline", "approved_fact"],
  "top_k": 8,
  "filters": {
    "status": "active"
  }
}
```

`brand_id` and `query` are required. `source_types` and `top_k` are optional; establish a documented default and maximum for `top_k`. The caller may issue multiple focused queries for separate passages or claims. Do not require the ingestion service to interpret a whole submitted document as one search query.

### Response

```json
{
  "brand_id": "brand_...",
  "results": [
    {
      "source_id": "src_...",
      "brand_id": "brand_...",
      "source_type": "guideline",
      "title": "Product naming",
      "version_id": "ver_...",
      "version": 1,
      "text": "Use the product name Trueline exactly as written.",
      "chunk_id": "chunk_...",
      "chunk_text": "Use the product name Trueline exactly as written.",
      "location": {
        "document_id": "doc_...",
        "document_name": "Brand guide.pdf",
        "section": "Product naming",
        "page": 4,
        "char_start": 1820,
        "char_end": 1878
      },
      "retrieval": {
        "method": "keyword",
        "rank": 1,
        "score": null
      }
    }
  ]
}
```

Requirements:

- Every result must include `brand_id`, `source_id`, `source_type`, `version_id`, and the exact retrieved `chunk_text`. Include the full source `text` when practical; otherwise provide an authoritative lookup by `source_id` + `version_id` for citation validation.
- `location` must refer to the original material and identify the chunk’s location where known. Use character offsets only when they are measured against a clearly defined canonical text representation.
- `retrieval.method` identifies the method actually used, such as `keyword`, `vector`, or `hybrid`. A score is optional and must not be presented as confidence that a source supports a claim.
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
- Keyword retrieval with a bounded result count.
- Stable source/version IDs and source locations where available.
- Explicit empty-result and failure behavior.
- A small fictional/demo dataset for local development. Do not commit real confidential brand material.

Keep embeddings, vector database selection, OCR/scanned documents, sophisticated ranking, and ingestion UI details out of the initial handoff unless the team has already approved them. The logical result contract should allow retrieval methods to evolve without changing how RAG validates citations.

## Acceptance checklist for the ingestion handoff

- [ ] Ingestion can create or load guideline and approved-fact records with stable brand/source/version identity.
- [ ] A source update produces a new immutable version.
- [ ] A query accepts a brand ID, focused text query, optional source types, and bounded `top_k`.
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
- When embeddings are introduced, which model and vector storage are approved, and how are model/index versions tracked?

Resolve these decisions with the backend and ingestion owners before treating this document as an implemented API. Update this contract when the actual implementation is agreed; keep it aligned with the code and API documentation.
