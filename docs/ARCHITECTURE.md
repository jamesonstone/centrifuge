# Centrifuge System Architecture

## Overview

Centrifuge is a deterministic CSV data cleaning pipeline that uses a rules-first approach with tightly-scoped LLM assistance. The system is designed as a distributed monolith with clear separation of concerns between API, worker, and storage layers.

### Architecture

```plaintext
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ API (8080)  │────▶│  PostgreSQL │◀────│  Workers    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                     │
       └───────────────────┴─────────────────────┘
                           │
                   ┌───────────────┐
                   │ MinIO Storage │
                   └───────────────┘

            [LLM via direct library calls]
```

## System Components

- **API Service** (`centrifuge-app`): FastAPI application handling run submissions and status queries (port 8080)
- **Worker Pool** (`centrifuge-worker-1/2`): Async workers that claim and process runs end-to-end
- **PostgreSQL Database**: Stores run state, canonical mappings cache, and artifact metadata
- **MinIO Storage**: S3-compatible object storage for input files and artifacts
- **LLM Integration**: Direct LiteLLM library integration (no proxy service in current implementation)

### 1. API Service (`centrifuge-app`)

- **Technology**: FastAPI with Uvicorn ASGI server
- **Port**: 8080
- **Responsibilities**:
  - Accept CSV file uploads (max 50,000 rows)
  - Create and queue runs
  - Serve run status and metadata
  - Provide artifact download endpoints
  - Health monitoring
- **Key Design**: Thin, non-blocking layer that never performs heavy processing
- **Why FastAPI?**: Async support, automatic OpenAPI docs, easy to containerize. Given more time, would consider Go or Rust for performance and static typing for the API layer and leverage python for "workers" only; but even then "workers", longer term, I'd build heterogeneously to maximize for the rules (deterministic) vs [python] LLM (non-deterministic) parts of the pipeline. Finally, the reason for python is that it has its roots in academia and data science, so it has the best libraries for data manipulation (pandas) and LLM integration (LangChain, LiteLLM, etc).

### 2. Worker Pool (`centrifuge-worker-1`, `centrifuge-worker-2`)

- **Technology**: Python asyncio with psycopg connection pooling
- **Scaling**: 2 workers by default, scalable to N
- **Responsibilities**:
  - Atomically claim runs from queue
  - Execute complete pipeline end-to-end
  - Send heartbeats every 30 seconds
  - Upload artifacts to MinIO
  - Update run status in database
- **Key Design**: Each worker owns a complete run; no phase distribution
- **Why asyncio?**: Lightweight concurrency model, easy to manage multiple I/O-bound tasks (DB, LLM, S3). Given more time, would consider a more robust task queue (e.g., Celery, RQ) or event-driven architecture (Kafka/Redpanda) for better scalability and fault tolerance. We're re-using the same image as the API service for simplicity, but in a more complex system I'd separate them.

### 3. PostgreSQL Database

- **Version**: Latest with pgvector extension
- **Schema**: Auto-initialized via docker-entrypoint-initdb.d
- **Tables**:
  - `runs`: Run metadata and state machine
  - `artifacts`: Artifact registry with content hashes
  - `canonical_mappings`: LLM mapping cache
  - `run_metrics`: Performance metrics per run
- **Key Design**: Single source of truth for all state
- **Why PostgreSQL?**: Reliable, ACID-compliant, supports complex queries. pgvector allows for future enhancements with vector search if needed. Given more time, would consider partitioning large tables or using a managed service for production reliability. Additionally, if the semantic search component were to be expanded or RAG architecture added, a dedicated vector database (e.g., Pinecone, Weaviate) might be warranted.

### 4. MinIO Object Storage [s3-compatible]

- **Protocol**: S3-compatible API
- **Bucket**: `artifacts` (auto-created)
- **Content**: Input files, cleaned CSVs, error reports, audit trails
- **Key Design**: Content-addressed storage using SHA-256 hashes
- **Why MinIO?**: Lightweight, easy to deploy locally, S3-compatible. In production, would use AWS S3 or another managed object storage service for durability and scalability.

### 5. LLM Integration

- **Library**: LiteLLM (direct integration, no proxy)
- **Model**: OpenAI gpt-4o (supports JSON response format)
- **Configuration**:
  - Temperature: 0.0 (deterministic)
  - Seed: 42 (reproducible)
  - Response format: JSON mode
- **Scope**: Only Department and Account Name columns
- **Key Design**: Strict contracts, edit caps, confidence thresholds
- **Why LiteLLM?**: Simplifies direct API calls, built-in retry logic. Given more time, would explore Anthropic Claude or Google Gemini for potential cost/performance benefits. Additionally, would consider fine-tuning a smaller open-source model (e.g., Llama 3) for on-premise deployments to reduce dependency on external APIs and improve data privacy. We set specific values to promote determinism and reproducibility, which are critical for auditability in data cleaning tasks. We also scope LLM usage narrowly to enum canonicalization to minimize costs and risks associated with non-deterministic outputs.

## Data Flow

```plaintext
1. Client uploads CSV to API
2. API stores file in MinIO, creates run record
3. Worker claims run atomically (FOR UPDATE SKIP LOCKED)
4. Worker executes pipeline phases:
   a. Ingest & profile CSV
   b. Apply deterministic rules (80%+ of fixes)
   c. Identify residuals for LLM processing
   d. Call OpenAI API for canonicalization
   e. Apply mappings with preconditions
   f. Final validation & quarantine
   g. Generate artifacts (7 files)
5. Worker updates run status to completed
6. Client retrieves artifacts via API
```

## Pipeline Phases

### Phase 1: Ingest & Validation

- Stream CSV with automatic delimiter detection
- Normalize headers and encoding
- Validate required columns and data types
- Compute content hash for idempotency

### Phase 2: Rules Engine

- **Deterministic transformations**:
  - Whitespace normalization
  - Case standardization (Title Case for text)
  - Date formatting (ISO 8601: YYYY-MM-DD)
  - Numeric cleaning and formatting
  - Transaction ID normalization (TXN-XXXXX format)
  - Debit/credit sign correction
- **Coverage**: Handles 80%+ of all fixes

### Phase 3: Residual Planning

- Identify values that rules couldn't fix
- Filter to LLM-eligible columns only
- Group by unique values for batching
- Check cache for existing mappings
- Apply 20% edit cap per column

### Phase 4: LLM Processing

- **Columns processed**: Department, Account Name only
- **Request batching**: Up to 50 unique values per call
- **Response validation**: Strict JSON contract
- **Confidence threshold**: 0.80 minimum
- **Caching**: Store successful mappings in PostgreSQL

### Phase 5: Final Validation

- Re-validate all business rules
- Check debit/credit consistency
- Verify transaction ID uniqueness
- Quarantine non-compliant rows with categories

### Phase 6: Artifact Generation

- **Files produced**:
  - `cleaned.csv`: Successfully processed rows
  - `errors.csv`: Quarantined rows with reasons
  - `diff.csv`: All changes made
  - `audit.ndjson`: Detailed change log
  - `manifest.json`: Run metadata and versions
  - `metrics.json`: Performance statistics
  - `summary.md`: Human-readable report
  - `input.csv`: Original uploaded file

## Determinism Guarantees

1. **Idempotency**: Content hash ensures identical inputs → identical outputs
2. **Reproducibility**: Fixed random seeds and temperature=0
3. **Preconditions**: Patches only apply if current value matches expected
4. **Version pinning**: All prompts and models versioned in manifest
5. **Audit trail**: Every change tracked with before/after values

## Scalability Design

### Current Implementation

- Database queue with atomic claiming
- Horizontal worker scaling (2-N workers)
- Connection pooling for database efficiency
- Content-addressed storage for deduplication

### Future Path

- Event bus (Kafka/Redpanda) for queue
- Distributed phase processing with downstream consumers
- WebSocket progress updates
- Multi-tenant isolation

## Security Considerations

1. **Data Exposure**:
   - Only enum columns sent to LLM
   - No PII in prompts (IDs, amounts excluded)
   - Description field retained but not sent to LLM

2. **API Security**:
   - Environment-based API keys
   - Mock mode for testing without credentials
   - Rate limiting via edit caps

3. **Storage Security**:
   - MinIO with access control
   - Content hashing prevents tampering
   - Artifact URLs include UUIDs

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db

# Storage
ARTIFACT_ENDPOINT=http://minio:9000
ARTIFACT_BUCKET=artifacts
ARTIFACT_ACCESS_KEY=minioadmin
ARTIFACT_SECRET_KEY=minioadmin

# LLM
OPENAI_API_KEY=sk-...
USE_MOCK_LLM=false
LLM_MODEL_ID=gpt-4o
LLM_TEMPERATURE=0
LLM_SEED=42

# Limits
EDIT_CAP_PCT=20
CONFIDENCE_FLOOR=0.80
```

### Key Parameters

- **Row limit**: 50,000 per file
- **Heartbeat interval**: 30 seconds
- **Visibility timeout**: 5 minutes
- **Max retries**: 3 for LLM calls
- **Worker poll interval**: 5 seconds

## Technology Stack

- **Language**: Python 3.13
- **API Framework**: FastAPI 0.115
- **Database**: PostgreSQL 16 with pgvector
- **Object Storage**: MinIO (S3-compatible)
- **LLM Library**: LiteLLM 1.77 (used as library although long term should be a proxy service-gateway)
- **Data Processing**: Pandas 2.2
- **Async Runtime**: asyncio with uvloop
- **Container Runtime**: Docker Compose
- **Package Manager**: uv (recommended) or pip

## Monitoring & Observability

### Health Endpoints

- `/health`: System health check
- Returns database and storage connectivity status

### Metrics Captured

- Rules vs LLM fix rates
- Processing time per phase
- Cache hit rates
- Quarantine reasons breakdown
- Token usage estimates

### Logging

- Structured logging with run_id correlation
- Separate streams for API and workers
- LLM operations heavily instrumented
- All errors include full context

## Testing Strategy

1. **Unit Tests**: Individual component validation
2. **Integration Tests**: End-to-end pipeline runs
3. **Concurrency Tests**: Worker claim atomicity
4. **Golden Tests**: Determinism verification
5. **Mock Mode**: Full pipeline without API calls

## Deployment

### Local Development

```bash
make stack-up        # Start all services
make demo-submit     # Submit test file
make stack-logs      # View logs
make stack-down      # Stop services
```

### Production Considerations

- Use managed PostgreSQL for reliability
- Configure S3 instead of MinIO
- Set up proper API key management
- Enable monitoring and alerting
- Configure backup strategies

## Limitations

1. **Current**:
   - 50,000 row limit per file
   - 2 LLM-enabled columns only
   - No real-time progress updates
   - Single-tenant architecture

2. **By Design**:
   - No automatic schema inference
   - No free-text generation
   - Strict edit caps (20%)
   - Deterministic processing only

## Future Enhancements

1. **Near Term**:
   - WebSocket progress updates
   - Batch file processing
   - Extended canonical lists
   - RAG for mapping suggestions

2. **Long Term**:
   - Multi-tenant isolation
   - Custom model fine-tuning
   - Human-in-the-loop workflows
   - Streaming data support
