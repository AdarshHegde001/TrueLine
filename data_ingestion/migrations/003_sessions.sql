-- Persist the currently active, ephemeral knowledge session.
CREATE TABLE IF NOT EXISTS knowledge_sessions (
    session_id       UUID PRIMARY KEY,
    organization_id  TEXT NOT NULL UNIQUE,
    session_brand_id TEXT NOT NULL UNIQUE,
    session_slot     BOOLEAN NOT NULL DEFAULT TRUE UNIQUE,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
