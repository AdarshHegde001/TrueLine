# Trueline RAG Review Workflow

**Status: proposed implementation workflow, aligned with the ingestion contract and Round 1 plan.**

This document describes what happens when a creator submits content for review. The ingestion workstream separately prepares approved, versioned brand knowledge and indexes it in PostgreSQL + pgvector using BGE-M3. The RAG/review pipeline consumes that knowledge; it does not ingest or rewrite the brand source of truth.

## Goal

For one submitted document and one selected brand, return a structured set of findings that are tied to exact submitted-content spans and, for violations, valid approved evidence. Retrieval supports the AI with context; it does not prove a violation by itself. No evidence means `unverified`, not false. AI output is untrusted until validated by ordinary application code.

## End-to-end flow

```text
Review request
  → validate request and brand context
  → parse submitted content and preserve locations
  → run deterministic checks ─────────────────────────┐
  → identify claim/tone passages and retrieve evidence ├→ evaluate with bounded AI steps
  → validate quotes, sources, versions, and findings ──┘
  → return review result to backend/frontend
  → creator/reviewer makes the decision
```

Deterministic checks and retrieval can run independently after parsing. Combine their outputs before relevant AI calls. The workflow is fixed and bounded; no model chooses arbitrary tools or changes application state.

## Stage-by-stage contract

### 1. Accept a review request

Input from the backend/API:

- `submission_id` and immutable `submission_version_id`
- `brand_id` and the already-authorized caller/brand context
- submitted text or a reference to the uploaded file
- content type and selected channel, when relevant
- supported language/configuration, if known

The RAG service must not treat a `brand_id` supplied by a caller as authorization. The backend enforces access. Retrieval still filters every query by that brand ID to prevent cross-brand results.

Reject malformed requests and unsupported file types visibly. Do not silently process a partial file as if parsing succeeded.

### 2. Parse the submitted content

The parser extracts text from the submitted content and produces a canonical text representation plus source locations. For the first useful flow, support plain text and text-based PDF/DOCX according to the agreed parser dependencies. Preserve page/paragraph/character location when available; never invent a location.

Each parsed unit should retain:

- exact extracted text
- stable order within the submission version
- location metadata (page, paragraph, character offsets, etc. when available)
- parser name/version and warnings

Character offsets are zero-based Python Unicode code-point offsets into the exact `ParsedDocument.text` returned by the parser. The frontend must convert offsets if its highlighting API uses a different string-indexing convention.

The original uploaded artifact remains unchanged. Parsing is for analysis, not a rewrite. A parse failure or empty extraction must be reported as a review error or human-actionable state, not as a clean pass.

### 3. Apply the privacy gate before model calls

Use the backend-approved privacy/masking policy before any model processing that requires masking, especially before an optional external model call. Preserve a reversible mapping only if the approved privacy design permits it. Record warnings/redaction state as required by the backend. Masking is a risk reduction, not a guarantee of anonymization.

Submitted content and retrieved source text are untrusted data. Instructions embedded in either must never change system policy, permissions, or tool access.

### 4. Run deterministic checks

Run code-only checks for objectively testable rules, such as preferred/banned terms and required disclaimers. Load the applicable, admin-confirmed structured rules through ingestion's deterministic-rule interface, then map them with `rag.rules.rules_from_ingestion_records`. A deterministic violation must retain the rule/source ID and version that triggered it. Never infer executable deterministic rules from free-form prose.

Deterministic checks do not call the LLM. A check that did not run or lacks applicable rules must not be reported as passed. A clean deterministic result means only that those implemented checks found no issue; it does not establish that the whole document is factually correct.

### 5. Build focused retrieval queries

Split the parsed document into review units (sentences/paragraphs or claim candidates) while retaining their original spans. Create focused queries for:

- claims that need comparison with approved facts
- relevant tone or messaging passages that need comparison with brand-voice rules
- any other explicitly supported review type

Do not submit the entire document as one undifferentiated query by default. Avoid unnecessary duplicate queries by deduplicating equivalent passages/queries. If no relevant query can be made, skip retrieval for that check and do not invent evidence.

### 6. Retrieve approved evidence

Call the ingestion/retrieval contract for each focused query with `brand_id`, query text, optional `source_types`, optional `document_types`, and bounded `top_k`. A specialist agent can filter to `tone_of_voice`, `approved_claims`, or `product_facts`; an empty filter searches all available types. The agreed target is hybrid PostgreSQL full-text + pgvector search, with BGE-M3 dense vectors (1024 dimensions) for semantic retrieval. The document type is prefixed into source chunk embedding input and returned as result metadata; exact stored passages remain unchanged. The query embedding must use the same model revision/configuration as the indexed chunks.

Suggested agent scopes once orchestration is added: tone review searches `tone_of_voice`, `brand_guidelines`, and `glossary`; claim verification searches `approved_claims` and `product_facts`, with `brand_guidelines` / `legal_compliance` when it needs claim-policy requirements. A typed category narrows retrieval but does not establish that a passage supports a finding; the citation and evidence checks still apply.

Every retrieved result must retain:

- `brand_id`, `source_id`, `version_id`, `source_type`, `document_type`
- exact `chunk_text` and source text or a reliable source/version lookup
- chunk ID and source location when available
- retrieval method/rank/score metadata

Only approved, active knowledge for the requested brand may be used. A retrieval score is not confidence and does not prove the passage supports a finding. Empty results are a valid “no matching approved evidence” response; database/index failures are errors and must not be disguised as empty results.

### 7. Evaluate with bounded AI components

For the Round 2 MVP, use the plan’s first three AI responsibilities as separate constrained calls or clearly separated steps in the fixed orchestrator:

- **A1 Tone Judge:** Compare the submission passage only with retrieved tone/voice guidelines. Do not invent tone standards.
- **A2 Claim Agent:** Extract a potentially verifiable claim and compare it with retrieved approved facts and applicable claim rules. Return `supported`, `contradicted`, or `unverified` for claim assessment. Lack of a matching fact is not evidence that the claim is false.
- **A3 Suggestion Agent:** Given a validated finding and relevant source context, propose a minimal change and optionally a rewrite. Suggestions must not introduce unsupported factual claims; they are proposals, not approved facts.

Each call receives only the relevant submission passage and small retrieved context, plus a strict structured-output schema. Use the selected local model/runtime after it is verified. If the model is unavailable or returns malformed output, preserve deterministic results and mark AI checks as failed/pending; do not fabricate successful findings.

### 8. Validate and assemble findings in code

Before a finding can be returned or stored, deterministic validators must check:

1. Output conforms to the agreed Pydantic finding schema.
2. The finding’s quote is an exact substring of the parsed submitted text and its span points to that quote.
3. Any cited source ID/version exists, belongs to the requested brand, is approved, and is the revision returned by retrieval.
4. The cited excerpt exactly matches the source/chunk text for that version.
5. A finding labeled `violation` has a valid relevant guideline/fact citation. Invalid or unsupported violations are rejected or downgraded to `unverified`/human review.
6. Suggestions are rechecked against applicable deterministic rules before display. If a suggestion still violates a deterministic rule, reject it or mark it as not verified; do not silently repair it.

Citation identity and exact-quote checks do not establish semantic entailment. For the MVP, the AI assessment is still a proposal and the human reviewer remains the final authority. The separate Critic agent (A4) and bounded verify-then-retry loop are later phases; code-level schema and citation validation are required from the start.

### 9. Return the review result

Return a structured response to the backend/frontend containing:

- submission ID/version and brand ID
- status of parsing, deterministic checks, retrieval, and AI steps
- findings with exact quote/span, label, severity, detector, explanation, confidence label, source ID/version/excerpt/location, suggestions, and validation status
- review/run metadata sufficient to audit model/configuration and retrieval method
- warnings/errors and checks that were skipped or could not complete

The `passed` label applies to a specific check that ran and found no issue. Do not interpret it as a blanket factual guarantee for the whole submission. Keep the unverified state explicit.

### 10. Human decision stays outside the RAG pipeline

RAG returns findings and proposed edits. It does not approve, publish, overwrite, or persistently modify the user’s draft. The backend workflow records accept/edit/dispute/escalate actions and applies an edit only after an authorized human action, with a diff and audit entry.

## MVP execution order

1. Agree and implement shared typed request/response/finding schemas.
2. Implement a parser result model and plain-text parsing path with spans; add text-based PDF/DOCX parsing in the agreed scope.
3. Implement deterministic validation utilities for quotes, spans, citations, brands, and source versions.
4. Implement a retrieval client/adapter matching `search_knowledge` in `data_ingestion/RAG_INGESTION_CONTRACT.md`; make it possible to inject a local fake only in tests/dev, never as hidden production fallback.
5. Add focused query construction and evidence linking.
6. Add deterministic terminology/disclaimer checks.
7. Add A1, A2, and A3 behind explicit model configuration and validate structured output.
8. Integrate the fixed orchestration path with the backend and return structured findings.
9. Measure retrieval and finding quality on the labelled evaluation set; track retrieval quality separately from generation/finding quality.

Do not start with a free-form agent, broad framework, or all six agents. First make parsing, retrieval, source linking, and validators work predictably with a small synthetic brand dataset.

## Failure behavior

| Failure | Required behavior |
|---|---|
| Unsupported/corrupt document or extraction yields no usable text | Return parse failure/warning; do not report a clean pass |
| Retrieval returns no matching sources | Continue deterministic checks; AI may label the claim unverified; do not call it false |
| PostgreSQL/pgvector or ingestion retrieval service is unavailable | Report retrieval failure; retain any independent deterministic results; no fabricated evidence |
| Model unavailable | Return available deterministic findings; mark AI checks unavailable/pending |
| Malformed model output | Reject it; bounded repair may be added later, but invalid output never reaches the user as a finding |
| Citation/source/span validation fails | Reject or downgrade the finding and record validation failure |
| Suggested text triggers deterministic rule | Do not present it as verified; retry is bounded in the later Critic milestone |

## Traceability and evaluation

Keep a per-review trace with submission version, parser status/version, checks run, query text or safe query reference, retrieval method, retrieved source/version/chunk IDs, model identifier/revision, structured output validation result, and timing/error metadata. Do not log secrets or unmasked confidential content contrary to the privacy policy.

Evaluate at least these separately:

- retrieval: whether relevant approved sources appear in top-k (recall/MRR or a small labelled retrieval set)
- citation validity: whether IDs, versions, and exact excerpts resolve
- semantic support: whether cited evidence actually supports the finding (human-labelled audit)
- finding quality: precision/recall on labelled cases, including clean and unverified cases
- failure behavior: no cross-brand results, no fake success on dependency/model failures, and no malformed outputs returned

Record measured outcomes and targets separately in `TESTING.md` once that project document and evaluation set exist.

## First coding milestone

The reusable non-AI review preparation path is now implemented under `rag/`:

- `models.py` defines parsed-document, source-reference, finding, span, and rule-result models.
- `parsing/` dispatches UTF-8 text, text-based PDF, and DOCX parsing and retains canonical offsets plus page/paragraph/table locations where available.
- `rules/` implements modular literal terminology and required-text/disclaimer checks with channel applicability, plus an adapter for the structured ingestion records.
- `validation.py` checks that any finding quote exactly matches the parsed text at its half-open character span.
- `retrieval/` defines shared search request/response models, a contract adapter for ingestion callables, and focused passage queries with offsets. It does not implement SQL or BGE-M3; the ingestion teammate owns that search function.
- `context.py` assembles parsed text, locations, deterministic outcomes, retrieval evidence, and retrieval status into a bounded JSON payload with a separate fixed system instruction.
- `review.py` wires rule loading, deterministic checks, retrieval, and context assembly into `prepare_review(...)`; it does not call any model.
- `test.py` is a manual Gemini API smoke script. It sends a fixed synthetic prompt only; it does not send a submitted document or review context.

These modules do not call an LLM and can be developed before the teammate’s ingestion implementation is complete. Deterministic checks accept typed rules mapped from approved `rule_data` in `data_ingestion/RAG_INGESTION_CONTRACT.md`; they do not infer executable rules from free-form prose. RAG-side retrieval orchestration and context assembly are implemented against the shared contract, but end-to-end retrieval is not live until ingestion supplies `search_knowledge` and `get_deterministic_rules`. AI/LangGraph orchestration remains later work.

Parser limitations at this stage: PDF OCR is not implemented; the parser returns a warning for pages with no extractable text. File-size limits, upload authorization, privacy masking, persistence, and backend API integration belong to their agreed platform boundaries and are not implemented in these RAG modules.
