# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Centrifuge is a deterministic CSV data cleaning pipeline with rules-first processing and tightly-contracted LLM assistance. It prioritizes auditability, transparency, and determinism over AI magic.

### Key Design Principles
- **Deterministic**: Identical inputs always produce identical outputs (temperature=0, seed=42)
- **Rules-First**: 80%+ of issues fixed deterministically via rules engine
- **Contracted LLM**: LLM only processes Department and Account Name columns with strict JSON contracts
- **Complete Audit Trail**: Every transformation tracked with source, reason, and confidence scores
- **Idempotent**: Re-running produces no additional changes

## Common Development Commands

### Stack Management
```bash
# Start complete stack with Docker Compose (includes Postgres, MinIO, LiteLLM)
make stack-up

# Start infrastructure only
make infra-start

# Stop everything
make stack-down

# View stack status
make stack-status

# Check system health
make health-check

# View logs
make stack-logs
```

### Development
```bash
# Run API in development mode
make api-start

# Start worker processes
make worker-start

# Scale workers (default 4)
make worker-scale WORKERS=4

# Run tests
make test-all           # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-concurrency  # Concurrency tests
make test-cov         # With coverage report

# Code quality
make lint              # Run linters
make format           # Format code with black/isort
make check            # Lint + test
```

### Database Operations
```bash
# Apply migrations
make db-migrate

# Database console
make db-console

# Backup database
make db-backup

# Restore database
make db-restore FILE=backup.sql

# Check database status
make db-status
```

### Monitoring & Debugging
```bash
# Check queue status
make queue-status

# Check worker status
make worker-status

# Cache statistics
make cache-stats

# Reset stale runs
make reset-stale-runs

# Debug environment variables
make debug-env
```

### Demo
```bash
# Run interactive demo
make demo

# Submit test run
make demo-submit

# Check recent runs
make demo-status

# Clean demo data
make demo-clean
```

## Architecture Overview

### System Components

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│     API     │────▶│  PostgreSQL │◀────│   Workers   │
│  (FastAPI)  │     │  (Metadata) │     │ (Async Pool)│
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                     │
       └───────────────────┴─────────────────────┘
                           │
                   ┌───────────────┐
                   │     MinIO     │
                   │  (Artifacts)  │
                   └───────────────┘
                           │
                   ┌───────────────┐
                   │    LiteLLM    │
                   │  (LLM Proxy)  │
                   └───────────────┘
```

### Data Flow & Phases

1. **Ingest** → 2. **Validation** → 3. **Rules Engine** → 4. **Residual Planning** → 5. **LLM Processing** → 6. **Apply Patches** → 7. **Re-validation** → 8. **Quarantine** → 9. **Artifacts**

### Key Directories

```
centrifuge/
├── api/                 # FastAPI application
│   ├── main.py         # Main API entry point
│   └── models.py       # API request/response models
├── worker/             # Async worker processes
│   └── main.py        # Worker process implementation
├── core/              # Business logic
│   ├── pipeline.py    # Main pipeline orchestrator
│   ├── rules.py       # Deterministic rules engine
│   ├── llm_adapter.py # LLM integration
│   ├── validation.py  # Schema validation
│   ├── quarantine.py  # Quarantine manager
│   └── artifacts.py   # Artifact generation
├── tests/             # Test suites
│   ├── test_rules.py
│   ├── test_worker_concurrency.py
│   └── test_integration.py
├── ops/sql/init/      # Database migrations
└── docker-compose.yaml # Infrastructure definition
```

## Development Patterns

### Database Connection Management
The project uses PostgreSQL with connection pooling through `psycopg`. All database operations go through `core/database.py`:

```python
from core.database import get_database, RunManager

db_pool = await get_database()
run_manager = RunManager(db_pool)
run_id = await run_manager.create_run(...)
```

### Worker Pattern
Workers use atomic claiming with `FOR UPDATE SKIP LOCKED`:
- Claim runs atomically
- Send heartbeats every 30 seconds
- 5-minute visibility timeout for stale runs
- Process entire run end-to-end (no phase distribution)

### LLM Integration
LLM is strictly scoped to two columns with JSON contracts:
- Temperature: 0 (deterministic)
- Seed: 42 (reproducible)
- Edit cap: 20% max changes
- Confidence floor: 0.80 minimum

### Storage Pattern
Artifacts use content-addressed storage with SHA-256 hashing:
- Input files and artifacts stored in MinIO
- Deduplication via content hashing
- Seven artifact types: cleaned, errors, diff, audit, manifest, metrics, summary

## Testing Strategy

### Test Categories
1. **Unit Tests** (`tests/test_*.py`): Individual component testing
2. **Integration Tests** (`test_integration.py`): End-to-end pipeline testing
3. **Concurrency Tests** (`test_worker_concurrency.py`): Worker claim atomicity
4. **Golden Tests**: Determinism verification with known inputs/outputs

### Running Single Tests
```bash
# Run specific test file
pytest tests/test_rules.py -v

# Run specific test function
pytest tests/test_rules.py::TestRulesEngine::test_date_normalization -v

# Run with debugging
pytest tests/test_integration.py -v -s --pdb
```

## Environment Configuration

### Required Environment Variables
```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/centrifuge

# Storage
ARTIFACT_ENDPOINT=http://localhost:9000
ARTIFACT_BUCKET=artifacts
ARTIFACT_ACCESS_KEY=minioadmin
ARTIFACT_SECRET_KEY=minioadmin

# LLM (for production)
OPENAI_API_KEY=sk-...
LLM_BASE_URL=http://localhost:4000
LLM_MODEL_ID=openai/gpt-4

# Worker Configuration
WORKER_ID=worker-1
HEARTBEAT_INTERVAL=30
VISIBILITY_TIMEOUT=300
POLL_INTERVAL=5

# Development Flags
USE_LOCAL_STORAGE=false  # Use MinIO vs local filesystem
USE_MOCK_LLM=false      # Use mock LLM for testing
```

## Troubleshooting Guide

### Common Issues

1. **Database Connection Failures**
   ```bash
   # Check database status
   docker-compose ps postgres
   docker-compose logs postgres
   # Restart if needed
   docker-compose restart postgres
   ```

2. **Worker Stalls**
   ```bash
   # Check for stale runs
   make worker-status
   # Reset stale runs
   make reset-stale-runs
   ```

3. **LLM API Failures**
   ```bash
   # Check LiteLLM proxy
   docker-compose logs litellm
   # Enable mock mode for testing
   export USE_MOCK_LLM=true
   ```

4. **MinIO Storage Issues**
   ```bash
   # Check MinIO status
   docker-compose ps minio
   # Access MinIO console
   open http://localhost:9001  # minioadmin/minioadmin
   ```

## Critical Implementation Notes

### Determinism Requirements
- All randomness must use fixed seeds
- Timestamps must be mocked in tests
- LLM responses must use temperature=0
- Row ordering must be preserved

### Data Contracts
- Schema version must be explicitly specified
- All patches include preconditions (before_value)
- Quarantine reasons must use defined categories
- Artifact manifests must include version pinning

### Performance Constraints
- 50,000 row limit per run
- 20% edit cap for LLM changes
- 5-minute worker visibility timeout
- 30-second heartbeat interval

### Security Considerations
- No PII sent to LLM (only Department and Account Name)
- All inputs sanitized before processing
- Strict output validation on LLM responses
- Rate limiting on API endpoints

## Future Enhancements (Not Implemented)

The following are documented for future implementation:
- Kafka/Redpanda for event streaming
- Multi-tenant authentication
- RAG-based canonicalization
- Human-in-the-loop approval workflow
- Real-time progress WebSockets
- Distributed phase processing

## Key Files to Review

When understanding the codebase, start with:
1. `core/pipeline.py` - Main orchestration logic
2. `api/main.py` - API endpoints and request handling
3. `worker/main.py` - Worker process and claiming logic
4. `core/rules.py` - Deterministic transformation rules
5. `core/llm_adapter.py` - LLM integration and contracts
6. `docker-compose.yaml` - Infrastructure setup
7. `Makefile` - All operational commands

## Development Workflow

1. **Make changes** to relevant files
2. **Run tests** with `make test-all`
3. **Check linting** with `make lint`
4. **Format code** with `make format`
5. **Test locally** with `make stack-up` and `make demo`
6. **Review audit trail** in generated artifacts

## Support Resources

- Architecture Decision Record: `docs/ADR-001-rules-first-llm.md`
- Operational Runbook: `docs/RUNBOOK.md`
- API Documentation: http://localhost:8000/docs (when running)
- Implementation Spec: `.specs/v1_centrifuge.md`