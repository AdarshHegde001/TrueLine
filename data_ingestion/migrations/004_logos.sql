-- Session-scoped approved logo references used by the visual review node.
CREATE TABLE IF NOT EXISTS knowledge_logos (
    logo_id       TEXT PRIMARY KEY,
    brand_id      TEXT NOT NULL REFERENCES knowledge_sessions(session_brand_id) ON DELETE CASCADE,
    title         TEXT NOT NULL,
    original_name TEXT NOT NULL,
    content_type  TEXT NOT NULL,
    storage_ref   TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    visual_hash   TEXT NOT NULL DEFAULT '',
    ocr_text      TEXT NOT NULL DEFAULT '',
    width         INTEGER NOT NULL CHECK (width > 0),
    height        INTEGER NOT NULL CHECK (height > 0),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (brand_id, sha256)
);

ALTER TABLE knowledge_logos ADD COLUMN IF NOT EXISTS visual_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE knowledge_logos ADD COLUMN IF NOT EXISTS ocr_text TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS knowledge_logos_brand_created_idx
    ON knowledge_logos (brand_id, created_at);
