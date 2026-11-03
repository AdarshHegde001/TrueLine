ALTER TABLE review_comments ADD COLUMN IF NOT EXISTS finding_index INTEGER;
DO $$ BEGIN
    ALTER TABLE review_comments ADD CONSTRAINT review_comments_finding_index_nonnegative
        CHECK (finding_index IS NULL OR finding_index >= 0);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS review_finding_decisions (
    review_id UUID NOT NULL REFERENCES review_cases(review_id) ON DELETE CASCADE,
    finding_index INTEGER NOT NULL CHECK (finding_index >= 0),
    status TEXT NOT NULL CHECK (status IN ('open', 'approved', 'changes_requested')) DEFAULT 'open',
    decided_by TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (review_id, finding_index)
);
