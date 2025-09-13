# Migration Guide: V1 to V2

## Overview
The V2 implementation adds production-ready infrastructure integration that was missing in V1. The key difference is moving from in-memory state management to persistent database-backed operations.

## What Changed

### 1. API Service (`api/main.py`)
**V1 (In-Memory):**
- Used in-memory `RunManager` class
- State lost on restart
- Single instance only
- No worker coordination

**V2 (Database-Backed):**
- Database for all state management
- State persists across restarts
- Supports multiple API instances
- Coordinates with async workers
- Proper health checks for all components

### 2. Database Integration
**V1:**
- Database existed but wasn't used by the application
- Schema was defined but not integrated

**V2:**
- Full database integration with connection pooling
- Atomic run claiming with `FOR UPDATE SKIP LOCKED`
- Persistent canonical mapping cache
- Artifact metadata tracking

### 3. Worker Implementation
**V1:**
- Basic worker skeleton existed
- No real implementation

**V2:**
- Complete async worker with continuous polling
- Heartbeat mechanism for liveness
- Visibility timeout for automatic reclaim
- Graceful shutdown handling

### 4. Storage Integration
**V1:**
- Local file system only
- No artifact management

**V2:**
- MinIO S3-compatible storage
- Content-addressed artifact storage
- Fallback to local storage for development

### 5. LLM Integration
**V1:**
- Multiple LLM adapter attempts
- Complex prompt management

**V2:**
- Simplified LLM client (`core/llm_client.py`)
- Direct integration with LiteLLM proxy
- Response caching in database
- Mock mode for testing

### 6. Makefile
**V1:**
- Basic targets for development
- Limited operational commands

**V2:**
- Comprehensive operational targets
- Stack management
- Monitoring commands
- Database operations
- Health checks and troubleshooting

## File Changes

### Replaced Files:
- `api/main.py` - Now uses database-backed implementation
- `Makefile` - Comprehensive operational targets

### New Files:
- `core/database.py` - Database connection and managers
- `core/pipeline.py` - Complete pipeline orchestration
- `core/storage.py` - Storage backend abstraction
- `core/llm_client.py` - Simplified LLM client
- `worker/main.py` - Complete worker implementation
- `docs/RUNBOOK.md` - Operational documentation
- `tests/test_worker_concurrency.py` - Concurrency tests

### Backup Files (can be deleted):
- `api/main_v1_backup.py` - Original in-memory API
- `Makefile.v1_backup` - Original basic Makefile

## Breaking Changes

### Environment Variables
New required environment variables:
```bash
# Database (defaults work for docker-compose)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=centrifuge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Storage (defaults work for docker-compose)
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# LLM (required for production)
OPENAI_API_KEY=<your-key>
```

### API Changes
The API endpoints remain the same, but the backend implementation is different:
- Runs are now persisted in database
- Status checks query database
- Artifacts are stored in MinIO

### Docker Compose
The docker-compose services now properly integrate:
- API connects to database
- Workers claim runs from database
- All services share MinIO storage

## Migration Steps

If you have existing V1 deployment:

1. **Backup any important data**
   ```bash
   make db-backup
   ```

2. **Stop V1 services**
   ```bash
   docker-compose down
   ```

3. **Update code to V2**
   - Already done by the file replacements above

4. **Start V2 services**
   ```bash
   make stack-up
   ```

5. **Verify health**
   ```bash
   make health-check
   ```

## Testing the Migration

Run the comprehensive test suite:
```bash
# Unit tests
make test-unit

# Integration tests (requires services running)
make stack-up
make test-integration

# Concurrency tests
make test-concurrency

# Full test suite
make test-all
```

## Rollback Plan

If you need to rollback to V1:
```bash
# Restore V1 files
cp api/main_v1_backup.py api/main.py
cp Makefile.v1_backup Makefile

# Restart services
docker-compose down
docker-compose up -d
```

## Benefits of V2

1. **Production Ready**: Proper state management, error handling, and recovery
2. **Scalable**: Multiple workers and API instances supported
3. **Reliable**: Automatic failure recovery and run reclamation
4. **Observable**: Comprehensive logging and metrics
5. **Maintainable**: Clear separation of concerns and documented operations

## Next Steps

1. Delete backup files if migration successful:
   ```bash
   rm api/main_v1_backup.py
   rm Makefile.v1_backup
   ```

2. Configure production environment variables

3. Review the runbook: `docs/RUNBOOK.md`

4. Start using V2 features:
   - Scale workers: `make worker-scale WORKERS=8`
   - Monitor health: `make health-check`
   - Check queue: `make queue-status`