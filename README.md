# TrueLine local prototype

This prototype connects the existing Python ingestion/RAG code to an Express API and a React/Vite UI for one configured organization. The backend review path parses a submitted document, runs deterministic checks, retrieves focused evidence, then executes a fixed LangGraph flow: tone review → claim verification → final summary.

## Layout

```text
TrueLine/
├── backend/                 Express API and persistent Python bridge
│   ├── server.js            Upload/review endpoints
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
    ├── agentic/             Tone, claims, and summary LangGraph nodes
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
   pip install -r rag/requirements.txt -r data_ingestion/requirements.txt
   ```

3. Set up and start the backend:

   ```bash
   cd backend
   npm install
   [ -f .env ] || cp .env.example .env
   # Add a Gemini API key to each agent's separate key variable in backend/.env.
   npm run dev
   ```

   The Python worker applies the DB migration at startup and automatically uses the repository `.venv` if present. The local `.env` now selects real `BAAI/bge-m3` embeddings. The first model load downloads its dense checkpoint (about 2.27 GB); the full Hugging Face repository is about 4.59 GB. Existing sources indexed with fake vectors must be reindexed with BGE-M3 before semantic retrieval (`python -m trueline_ingestion reindex` from `data_ingestion/`, using the backend `.env` settings). Keep the same embedder/model for subsequent ingestion and searches.

4. In another terminal, start the frontend:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   The frontend uses `http://localhost:3001` by default; set `VITE_API_BASE_URL` in `frontend/.env` only if the API runs elsewhere. Open `http://localhost:5173`.

## Demo workflow

1. Start a session from the header or home page. A session creates a fresh, isolated knowledge scope. The API requires its session token for knowledge uploads, source listing, and reviews.
2. In **Knowledge library**, assign a document type to each `.txt`, `.md`, `.pdf`, or `.docx` file (15 MB maximum) and submit the batch. The UI waits for all uploads to finish embedding and indexing before allowing review.
3. For deterministic checks, structured rules must be supplied through the ingestion contract. The system does not infer executable rules from prose.

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

4. In **Document review**, submit the verification document and select a channel. The backend parses it, runs deterministic checks, retrieves general and specialist evidence, and executes three configured model calls in order.
5. The tone agent checks voice guidance; the claims agent checks approved claims/product facts and receives the tone result; the summarizer receives all retrieved evidence, deterministic findings, and both prior agent results.
6. Choose **End session** when finished and confirm. The backend deletes the knowledge sources, versions, chunks, and embeddings before allowing the next session to start.

## Agent model settings

All three agents use Gemini through its OpenAI-compatible endpoint, with a separate key variable for each agent. Set `TRUELINE_TONE_API_KEY`, `TRUELINE_CLAIMS_API_KEY`, and `TRUELINE_SUMMARY_API_KEY` in `backend/.env`. The endpoint is `https://generativelanguage.googleapis.com/v1beta/openai/`; the current test model is `gemini-3.1-flash-lite`. Google documents this endpoint for OpenAI-compatible clients in its [Gemini API guide](https://ai.google.dev/gemini-api/docs/openai).

The provider, model, endpoint, key, and timeout are configured independently for tone, claims, and summary. Keep the keys private; they are sent only from the backend to Gemini and are never included in health or review responses.

## Current prototype limits

- This is a local prototype; the session token isolates temporary data but is not user authentication or an authorization system. Do not expose the API to the public internet.
- Deterministic rules must be explicitly entered/reviewed as structured JSON; prose extraction is not implemented.
- Agent outputs are proposals, not compliance decisions; a human must review findings.
- Scanned PDFs need OCR, which is not implemented.
- Integration tests need a disposable test DB because their fixture drops `knowledge_*` tables. Follow `data_ingestion/README.md` and use its `trueline_test` database.
