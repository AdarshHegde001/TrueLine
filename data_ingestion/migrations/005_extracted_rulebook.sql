-- Model-extracted, evidence-linked guidance is stored separately from the
-- machine-checkable rule_data contract.
ALTER TABLE knowledge_source_versions
    ADD COLUMN IF NOT EXISTS extracted_rules JSONB NOT NULL DEFAULT '[]'::jsonb;

DO $$ BEGIN
    ALTER TABLE knowledge_source_versions
        ADD CONSTRAINT knowledge_source_versions_extracted_rules_array
        CHECK (jsonb_typeof(extracted_rules) = 'array');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
