# TrueLine local prototype

This prototype connects the existing Python ingestion/RAG code to an Express API and a React/Vite UI for one configured organization. The UI has two flows: add typed organization knowledge, then submit a document for deterministic checks, evidence retrieval, and a Gemini smoke-test review.

## Layout

```text
TrueLine/
├── backend/                 Express API and persistent Python bridge
│   ├── server.js            Upload/review endpoints and Gemini request
│   ├── python_worker.py     Calls ingestion and RAG Python entry points
│   ├── package.json         Backend dependencies and run scripts
│   └── .env.example         Local environment template; keep .env private
├── frontend/                React + Vite application
│   └── src/
│       ├── App.jsx          Knowledge upload and document review screens
│       ├── main.jsx         React entry point
│       └── style.css        Application styles
├── data_ingestion/          PostgreSQL/pgvector knowledge ingestion
│   ├── migrations/          Database schema migrations
│   ├── seed/                Optional demo records
│   ├── tests/               Ingestion and retrieval tests
│   └── trueline_ingestion/  Source management, embedding and search code
└── rag/                     Document parsing and review preparation
    ├── parsing/             PDF, DOCX and text parsers
    ├── retrieval/           Ingestion adapter and focused retrieval
    └── rules/               Deterministic rule models and checks
```

## Run locally

Use Python 3.11 or 3.12 for the current ML packages.

1. Start PostgreSQL + pgvector:

   ```bash
   (cd data_ingestion && docker compose up -d db)
   ```

   After the document-type embedding format upgrade, existing approved sources need reindexing once with the same embedder configured in `backend/.env` (`python -m trueline_ingestion reindex` from `data_ingestion/`; for an existing fake-indexed demo, add `--fake`).

2. Install the Python dependencies for the quick demo (the fake embedder avoids the large BGE model download):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install "psycopg[binary]>=3.2" "pydantic>=2.7" "PyMuPDF>=1.24" "python-docx>=1.1"
   ```

3. Set up and start the backend:

   ```bash
   cd backend
   npm install
   [ -f .env ] || cp .env.example .env
   # Add GEMINI_API_KEY to backend/.env; keep this file private.
   npm run dev
   ```

   The Python worker applies the DB migration at startup and automatically uses the repository `.venv` if present. The checked-in `.env.example` and local `.env` use `INGESTION_EMBEDDER=fake` to make the first plumbing demo quick. Fake embeddings are only for testing request flow and must not be used to evaluate semantic retrieval. To use BGE-M3, install `sentence-transformers>=3.0`, set `INGESTION_EMBEDDER=bge-m3`, and restart the backend; the first model load downloads the model. All knowledge used in one index must be embedded and searched with the same mode/model, so after switching, reindex the existing knowledge (`python -m trueline_ingestion reindex` from `data_ingestion/`, with the same environment).

4. In another terminal, start the frontend:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   The frontend uses `http://localhost:3001` by default; set `VITE_API_BASE_URL` in `frontend/.env` only if the API runs elsewhere. Open `http://localhost:5173`.

## Demo workflow

1. In **Knowledge library**, select the document type and upload a `.txt`, `.md`, `.pdf`, or `.docx` file (15 MB maximum). Supported types are brand guidelines, tone of voice, approved claims, product facts, glossary, asset rules, legal & compliance, and other reference.
2. For deterministic checks, paste explicitly reviewed structured rules into the optional JSON array. The system does not infer executable rules from prose. Example:

   ```json
   [{
     "rule_id": "rule_name_1",
     "kind": "terminology",
     "mode": "replace_with_preferred",
     "term": "Tru Line",
     "replacement": "Trueline",
     "case_sensitive": false,
     "whole_word": true,
     "severity": "medium",
     "applies_to_channels": []
   }]
   ```

3. In **Document review**, submit the verification document and select a channel. The backend uses the configured organization, parses the document, runs deterministic rules, retrieves evidence, assembles RAG context, and sends that context to Gemini if `GEMINI_API_KEY` is set.
4. Inspect the deterministic findings, Gemini response, and expandable JSON context in the UI.

## Gemini settings

`backend/.env` accepts `TRUELINE_BRAND_ID` and `TRUELINE_ORGANIZATION_NAME` for the single organization, plus `GEMINI_API_KEY`, `GEMINI_API_BASE_URL`, and `GEMINI_MODEL`. The browser no longer sends or edits a brand ID; both ingest and review use the backend-configured organization. The default Gemini model is `gemini-3.1-flash-lite`; the configured base URL is `https://generativelanguage.googleapis.com/v1beta`. Google’s current pricing page lists free-tier input/output for this model, subject to quota and availability. The Gemini free tier may use submitted prompts to improve Google products, so use non-sensitive test documents unless your data policies allow otherwise. See the [Gemini model list](https://ai.google.dev/gemini-api/docs/models), [pricing](https://ai.google.dev/gemini-api/docs/pricing), and [Generate Content API](https://ai.google.dev/api/generate-content).

If the Gemini key is blank, RAG still runs and the UI shows the assembled context, while the AI section reports that Gemini is not configured. The key stays in the backend environment and is never sent to the browser.

## Current prototype limits

- This is a local prototype; it has no authentication or user/brand authorization layer. Do not expose the API to the public internet.
- Deterministic rules must be explicitly entered/reviewed as structured JSON; prose extraction is not implemented.
- The smoke-test response is not a compliance decision. A human must review findings.
- Scanned PDFs need OCR, which is not implemented.
- Integration tests need a disposable test DB because their fixture drops `knowledge_*` tables. Follow `data_ingestion/README.md` and use its `trueline_test` database.
