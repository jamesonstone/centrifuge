-- Centrifuge Database Schema
-- PostgreSQL 15+ with pgvector extension
-- Uses UUIDv7 for primary keys (time-sortable UUIDs)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create custom domain for UUIDv7 (simulated with gen_random_uuid for PoC)
-- Note: In production, use a proper UUIDv7 function
CREATE OR REPLACE FUNCTION generate_uuidv7() RETURNS uuid AS $$
BEGIN
    RETURN gen_random_uuid();
END;
$$ LANGUAGE plpgsql;

-- Enum types
CREATE TYPE run_status AS ENUM ('queued', 'running', 'succeeded', 'partial', 'failed');
CREATE TYPE source_type AS ENUM ('rule', 'llm', 'human', 'cache');
CREATE TYPE error_category AS ENUM (
    'validation_failure',
    'llm_contract_failure', 
    'low_confidence',
    'edit_cap_exceeded',
    'parse_error'
);

-- =====================================================================
-- RUNS TABLE - Core run tracking
-- =====================================================================
CREATE TABLE runs (
    id UUID PRIMARY KEY DEFAULT generate_uuidv7(),
    run_seq BIGSERIAL UNIQUE NOT NULL, -- human-friendly sequential ID
    
    -- Input tracking
    input_file_name VARCHAR(255),
    input_file_hash VARCHAR(64) NOT NULL, -- SHA256 of input file
    input_row_count INTEGER,
    
    -- Status and progress
    status run_status NOT NULL DEFAULT 'queued',
    phase_progress JSONB DEFAULT '{"phase": "queued", "percent": 0}'::jsonb,
    
    -- Options
    options JSONB NOT NULL DEFAULT '{}'::jsonb,
    use_inferred BOOLEAN DEFAULT FALSE,
    dry_run BOOLEAN DEFAULT FALSE,
    llm_columns TEXT[] DEFAULT ARRAY['Department', 'Account Name'],
    
    -- Worker assignment
    worker_id VARCHAR(50),
    claimed_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    
    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    -- Results summary
    total_rows INTEGER,
    cleaned_rows INTEGER,
    quarantined_rows INTEGER,
    rules_fixed_count INTEGER,
    llm_fixed_count INTEGER,
    error_breakdown JSONB DEFAULT '{}'::jsonb,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for runs
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_worker ON runs(worker_id, status);
CREATE INDEX idx_runs_created ON runs(created_at DESC);
CREATE INDEX idx_runs_heartbeat ON runs(heartbeat_at) WHERE status = 'running';

-- =====================================================================
-- SCHEMAS TABLE - Store schema definitions
-- =====================================================================
CREATE TABLE schemas (
    id UUID PRIMARY KEY DEFAULT generate_uuidv7(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    
    -- Schema definition
    columns JSONB NOT NULL,
    constraints JSONB DEFAULT '{}'::jsonb,
    header_aliases JSONB DEFAULT '{}'::jsonb,
    domain_rules JSONB DEFAULT '{}'::jsonb,
    
    -- Canonical values
    canonical_departments TEXT[] DEFAULT ARRAY[
        'Sales', 'Operations', 'Admin', 'IT', 'Finance', 
        'Marketing', 'HR', 'Legal', 'Engineering', 'Support'
    ],
    canonical_accounts TEXT[] DEFAULT ARRAY[
        'Cash', 'Accounts Receivable', 'Accounts Payable',
        'Sales Revenue', 'Cost of Goods Sold', 'Operating Expenses',
        'Equipment', 'Inventory', 'Retained Earnings'
    ],
    
    -- Metadata
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(name, version)
);

-- =====================================================================
-- CANONICAL_MAPPINGS TABLE - Cache for LLM decisions
-- =====================================================================
CREATE TABLE canonical_mappings (
    id UUID PRIMARY KEY DEFAULT generate_uuidv7(),
    
    -- Mapping key
    column_name VARCHAR(100) NOT NULL,
    variant_value TEXT NOT NULL,
    canonical_value TEXT NOT NULL,
    
    -- Versioning and source
    model_id VARCHAR(100) NOT NULL,
    prompt_version VARCHAR(50) NOT NULL,
    source source_type NOT NULL,
    confidence DECIMAL(3,2),
    
    -- Approval tracking
    is_approved BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR(100),
    approved_at TIMESTAMPTZ,
    
    -- Usage stats
    use_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    superseded_at TIMESTAMPTZ,
    superseded_by UUID REFERENCES canonical_mappings(id),
    
    -- Ensure uniqueness of active mappings
    UNIQUE(column_name, variant_value, model_id, prompt_version) 
        DEFERRABLE INITIALLY DEFERRED
);

-- Indexes for canonical_mappings
CREATE INDEX idx_mappings_lookup ON canonical_mappings(column_name, variant_value) 
    WHERE superseded_at IS NULL;
CREATE INDEX idx_mappings_approved ON canonical_mappings(is_approved) 
    WHERE superseded_at IS NULL;

-- =====================================================================
-- ARTIFACTS TABLE - Track all generated artifacts
-- =====================================================================
CREATE TABLE artifacts (
    id UUID PRIMARY KEY DEFAULT generate_uuidv7(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    
    -- Artifact identification
    artifact_type VARCHAR(50) NOT NULL, -- cleaned, errors, diff, audit, manifest, metrics, summary
    file_name VARCHAR(255) NOT NULL,
    
    -- Storage
    content_hash VARCHAR(64) NOT NULL, -- SHA256
    storage_path TEXT NOT NULL, -- MinIO path
    size_bytes BIGINT,
    
    -- Metadata
    mime_type VARCHAR(100),
    row_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for artifacts
CREATE INDEX idx_artifacts_run ON artifacts(run_id);
CREATE INDEX idx_artifacts_hash ON artifacts(content_hash);
CREATE INDEX idx_artifacts_type ON artifacts(run_id, artifact_type);

-- =====================================================================
-- AUDIT_LOG TABLE - Detailed change tracking
-- =====================================================================
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT generate_uuidv7(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    
    -- Change details
    row_uuid UUID NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    before_value TEXT,
    after_value TEXT,
    
    -- Source and reasoning
    source source_type NOT NULL,
    rule_id VARCHAR(100),
    contract_id VARCHAR(100),
    reason TEXT,
    confidence DECIMAL(3,2),
    
    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for audit_log
CREATE INDEX idx_audit_run ON audit_log(run_id);
CREATE INDEX idx_audit_row ON audit_log(row_uuid);
CREATE INDEX idx_audit_source ON audit_log(source);

-- =====================================================================
-- METRICS TABLE - Store run metrics
-- =====================================================================
CREATE TABLE metrics (
    id UUID PRIMARY KEY DEFAULT generate_uuidv7(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    
    -- Counts
    total_cells INTEGER,
    cells_validated INTEGER,
    cells_modified INTEGER,
    cells_quarantined INTEGER,
    
    -- Performance
    rules_duration_ms INTEGER,
    llm_duration_ms INTEGER,
    total_duration_ms INTEGER,
    
    -- LLM stats
    llm_calls_count INTEGER DEFAULT 0,
    llm_tokens_used INTEGER DEFAULT 0,
    llm_cost_estimate DECIMAL(10,4),
    cache_hits INTEGER DEFAULT 0,
    cache_misses INTEGER DEFAULT 0,
    
    -- Error breakdown by category
    validation_failures INTEGER DEFAULT 0,
    llm_contract_failures INTEGER DEFAULT 0,
    low_confidence_count INTEGER DEFAULT 0,
    edit_cap_exceeded_count INTEGER DEFAULT 0,
    parse_errors INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for metrics
CREATE INDEX idx_metrics_run ON metrics(run_id);

-- =====================================================================
-- HELPER FUNCTIONS
-- =====================================================================

-- Function to claim a run for processing
CREATE OR REPLACE FUNCTION claim_run(p_worker_id VARCHAR(50))
RETURNS TABLE(run_id UUID, input_file_hash VARCHAR, options JSONB) AS $$
BEGIN
    RETURN QUERY
    UPDATE runs
    SET 
        status = 'running',
        worker_id = p_worker_id,
        claimed_at = NOW(),
        heartbeat_at = NOW(),
        started_at = NOW(),
        updated_at = NOW()
    WHERE id = (
        SELECT id 
        FROM runs 
        WHERE status = 'queued'
        ORDER BY created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    RETURNING id, runs.input_file_hash, runs.options;
END;
$$ LANGUAGE plpgsql;

-- Function to update heartbeat
CREATE OR REPLACE FUNCTION update_heartbeat(p_run_id UUID, p_worker_id VARCHAR(50))
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE runs
    SET heartbeat_at = NOW()
    WHERE id = p_run_id AND worker_id = p_worker_id AND status = 'running';
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Function to get or create canonical mapping
CREATE OR REPLACE FUNCTION get_or_create_mapping(
    p_column_name VARCHAR(100),
    p_variant_value TEXT,
    p_canonical_value TEXT,
    p_model_id VARCHAR(100),
    p_prompt_version VARCHAR(50),
    p_source source_type,
    p_confidence DECIMAL(3,2)
)
RETURNS UUID AS $$
DECLARE
    v_mapping_id UUID;
BEGIN
    -- First try to find existing approved mapping
    SELECT id INTO v_mapping_id
    FROM canonical_mappings
    WHERE column_name = p_column_name
      AND variant_value = p_variant_value
      AND is_approved = TRUE
      AND superseded_at IS NULL
    ORDER BY created_at DESC
    LIMIT 1;
    
    IF v_mapping_id IS NOT NULL THEN
        -- Update use count
        UPDATE canonical_mappings
        SET use_count = use_count + 1,
            last_used_at = NOW()
        WHERE id = v_mapping_id;
        
        RETURN v_mapping_id;
    END IF;
    
    -- Create new mapping
    INSERT INTO canonical_mappings (
        column_name, variant_value, canonical_value,
        model_id, prompt_version, source, confidence
    ) VALUES (
        p_column_name, p_variant_value, p_canonical_value,
        p_model_id, p_prompt_version, p_source, p_confidence
    )
    ON CONFLICT (column_name, variant_value, model_id, prompt_version) 
    DO UPDATE SET
        use_count = canonical_mappings.use_count + 1,
        last_used_at = NOW()
    RETURNING id INTO v_mapping_id;
    
    RETURN v_mapping_id;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_runs_updated_at BEFORE UPDATE ON runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_schemas_updated_at BEFORE UPDATE ON schemas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================================
-- INITIAL DATA
-- =====================================================================

-- Insert default schema
INSERT INTO schemas (name, version, columns, constraints, domain_rules) VALUES (
    'accounting_v1',
    '1.0.0',
    '{
        "Transaction ID": {"type": "string", "required": true, "allow_in_prompt": false},
        "Date": {"type": "date", "required": true, "allow_in_prompt": false},
        "Account Code": {"type": "string", "required": true, "allow_in_prompt": false},
        "Account Name": {"type": "string", "required": true, "allow_in_prompt": true, "llm_enabled": true},
        "Description": {"type": "string", "required": false, "allow_in_prompt": false},
        "Debit Amount": {"type": "decimal", "required": false, "allow_in_prompt": false},
        "Credit Amount": {"type": "decimal", "required": false, "allow_in_prompt": false},
        "Department": {"type": "string", "required": true, "allow_in_prompt": true, "llm_enabled": true},
        "Reference Number": {"type": "string", "required": false, "allow_in_prompt": false},
        "Created By": {"type": "string", "required": false, "allow_in_prompt": false}
    }'::jsonb,
    '{"primary_key": "Transaction ID", "unique": ["Transaction ID"], "debit_xor_credit": true}'::jsonb,
    '{"debit_xor_credit": "Exactly one of Debit Amount or Credit Amount must be positive"}'::jsonb
);