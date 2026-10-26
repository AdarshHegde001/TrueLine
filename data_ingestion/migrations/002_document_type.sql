-- Persist the organization-selected document purpose at source-version and
-- chunk level so retrieval can filter by the future review agent's domain.
ALTER TABLE knowledge_source_versions
    ADD COLUMN IF NOT EXISTS document_type TEXT NOT NULL DEFAULT 'other_reference';

ALTER TABLE knowledge_chunks
    ADD COLUMN IF NOT EXISTS document_type TEXT NOT NULL DEFAULT 'other_reference';

DO $$ BEGIN
    ALTER TABLE knowledge_source_versions
        ADD CONSTRAINT knowledge_source_versions_document_type_check
        CHECK (document_type IN ('brand_guidelines', 'tone_of_voice', 'approved_claims',
            'product_facts', 'glossary', 'asset_rules', 'legal_compliance', 'other_reference'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE knowledge_chunks
        ADD CONSTRAINT knowledge_chunks_document_type_check
        CHECK (document_type IN ('brand_guidelines', 'tone_of_voice', 'approved_claims',
            'product_facts', 'glossary', 'asset_rules', 'legal_compliance', 'other_reference'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

UPDATE knowledge_source_versions
SET document_type = metadata ->> 'document_type'
WHERE document_type = 'other_reference'
  AND metadata ->> 'document_type' IN ('brand_guidelines', 'tone_of_voice', 'approved_claims',
      'product_facts', 'glossary', 'asset_rules', 'legal_compliance', 'other_reference');

UPDATE knowledge_chunks c
SET document_type = v.document_type
FROM knowledge_source_versions v
WHERE v.version_id = c.version_id
  AND c.document_type = 'other_reference';

CREATE INDEX IF NOT EXISTS idx_chunks_brand_document_type_status
    ON knowledge_chunks (brand_id, document_type, index_status);
