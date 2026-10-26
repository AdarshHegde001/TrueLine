-- Trueline data ingestion schema (see RAG_INGESTION_CONTRACT.md)
-- Idempotent: safe to run more than once.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_sources (
    source_id           TEXT PRIMARY KEY,
    brand_id            TEXT NOT NULL,
    source_type         TEXT NOT NULL CHECK (source_type IN ('guideline', 'approved_fact')),
    title               TEXT NOT NULL,
    current_version_id  TEXT,            -- FK added below (circular)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_id, brand_id)         -- target for composite FKs
);
CREATE INDEX IF NOT EXISTS idx_sources_brand ON knowledge_sources (brand_id);

CREATE TABLE IF NOT EXISTS knowledge_source_versions (
    version_id       TEXT PRIMARY KEY,
    source_id        TEXT NOT NULL,
    brand_id         TEXT NOT NULL,
    version_number   INTEGER NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('pending', 'active', 'inactive', 'archived')),
    approval_status  TEXT NOT NULL CHECK (approval_status IN ('approved', 'pending', 'rejected')),
    text             TEXT NOT NULL,
    rule_data        JSONB NOT NULL DEFAULT '[]'::jsonb,
    document_id      TEXT,
    document_name    TEXT,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_id, version_number),
    UNIQUE (version_id, source_id, brand_id),
    -- brand on a version must equal brand on its source
    FOREIGN KEY (source_id, brand_id) REFERENCES knowledge_sources (source_id, brand_id)
);
-- Keep this migration safe for databases created by the earlier schema draft.
ALTER TABLE knowledge_source_versions
    ADD COLUMN IF NOT EXISTS rule_data JSONB NOT NULL DEFAULT '[]'::jsonb;
DO $$ BEGIN
    ALTER TABLE knowledge_source_versions
        ADD CONSTRAINT knowledge_source_versions_rule_data_array
        CHECK (jsonb_typeof(rule_data) = 'array');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
CREATE INDEX IF NOT EXISTS idx_versions_brand_status ON knowledge_source_versions (brand_id, status);

DO $$ BEGIN
    ALTER TABLE knowledge_sources
        ADD CONSTRAINT fk_sources_current_version
        FOREIGN KEY (current_version_id) REFERENCES knowledge_source_versions (version_id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id          TEXT PRIMARY KEY,
    brand_id          TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    version_id        TEXT NOT NULL,
    chunk_index       INTEGER NOT NULL,
    chunk_text        TEXT NOT NULL,
    location          JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_tsv        TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    embedding         vector(1024),
    embedding_model   TEXT NOT NULL,
    chunking_version  TEXT NOT NULL,
    index_status      TEXT NOT NULL CHECK (index_status IN ('pending', 'ready', 'failed')),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (version_id, chunk_index),
    -- chunk brand/source must agree with the parent version
    FOREIGN KEY (version_id, source_id, brand_id)
        REFERENCES knowledge_source_versions (version_id, source_id, brand_id),
    -- only 'ready' chunks may be searchable, and ready requires a vector
    CHECK (index_status <> 'ready' OR embedding IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_chunks_brand_status ON knowledge_chunks (brand_id, index_status);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON knowledge_chunks USING GIN (search_tsv);

-- Exact (sequential) vector search is used until the dataset justifies an ANN index.
-- To enable HNSW later (document parameters + re-evaluate retrieval quality first):
--   CREATE INDEX idx_chunks_embedding_hnsw ON knowledge_chunks
--       USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
