-- Human review records for extracted rules and collaborative document reviews.
CREATE TABLE IF NOT EXISTS extracted_rule_reviews (
    review_id TEXT PRIMARY KEY,
    brand_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    rule_data JSONB NOT NULL CHECK (jsonb_typeof(rule_data) = 'object'),
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')) DEFAULT 'pending',
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    updated_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_version_id, rule_id)
);
CREATE INDEX IF NOT EXISTS idx_extracted_rule_reviews_brand ON extracted_rule_reviews (brand_id, status);

CREATE TABLE IF NOT EXISTS extracted_rule_review_audit (
    audit_id BIGSERIAL PRIMARY KEY,
    review_id TEXT NOT NULL REFERENCES extracted_rule_reviews(review_id),
    revision INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('created', 'edited', 'approved', 'rejected', 'reopened')),
    actor TEXT NOT NULL,
    prior_rule JSONB,
    next_rule JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS review_cases (
    review_id UUID PRIMARY KEY,
    session_id UUID NOT NULL,
    brand_id TEXT NOT NULL,
    document_name TEXT NOT NULL,
    findings JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(findings) = 'array'),
    assigned_to TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'in_review', 'approved', 'changes_requested')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_review_cases_session ON review_cases (session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS review_comments (
    comment_id BIGSERIAL PRIMARY KEY,
    review_id UUID NOT NULL REFERENCES review_cases(review_id) ON DELETE CASCADE,
    author TEXT NOT NULL,
    body TEXT NOT NULL CHECK (length(body) BETWEEN 1 AND 4000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_review_comments_case ON review_comments (review_id, created_at);

-- Backfill any extracted proposals created before this workflow was installed.
INSERT INTO extracted_rule_reviews
    (review_id,brand_id,source_id,source_version_id,rule_id,rule_data,status,revision,updated_by)
SELECT 'rr_' || md5(v.version_id || ':' || (item.rule_data->>'rule_id')),
       v.brand_id,v.source_id,v.version_id,item.rule_data->>'rule_id',item.rule_data,'pending',1,'system'
FROM knowledge_source_versions v
CROSS JOIN LATERAL jsonb_array_elements(v.extracted_rules) AS item(rule_data)
WHERE item.rule_data ? 'rule_id'
ON CONFLICT (source_version_id,rule_id) DO NOTHING;

INSERT INTO extracted_rule_review_audit (review_id,revision,action,actor,next_rule)
SELECT r.review_id,r.revision,'created','system',r.rule_data
FROM extracted_rule_reviews r
WHERE NOT EXISTS (SELECT 1 FROM extracted_rule_review_audit a WHERE a.review_id=r.review_id);
