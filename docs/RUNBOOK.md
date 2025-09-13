# Centrifuge Operational Runbook

## Table of Contents
- [System Overview](#system-overview)
- [Startup Procedures](#startup-procedures)
- [Shutdown Procedures](#shutdown-procedures)
- [Health Checks](#health-checks)
- [Common Failure Modes](#common-failure-modes)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Emergency Procedures](#emergency-procedures)

## System Overview

Centrifuge is a data cleaning pipeline that processes CSV files through deterministic rules and LLM-assisted canonicalization.

### Components
- **API Service**: FastAPI application handling run submissions and status queries
- **Worker Processes**: Async workers that claim and process runs
- **PostgreSQL Database**: Stores run state, canonical mappings cache, and artifact metadata
- **MinIO Storage**: S3-compatible object storage for input files and artifacts
- **LiteLLM Proxy**: Provides unified interface to LLM providers

### Architecture
```
┌─────────┐     ┌─────────┐     ┌──────────┐
│   API   │────▶│   DB    │◀────│  Worker  │
└─────────┘     └─────────┘     └──────────┘
     │               │                 │
     └───────────────┴─────────────────┘
                     │
                ┌─────────┐
                │  MinIO  │
                └─────────┘
                     │
                ┌─────────┐
                │ LiteLLM │
                └─────────┘
```

## Startup Procedures

### 1. Infrastructure Services
Start infrastructure services using docker-compose:

```bash
# Start all infrastructure services
docker-compose up -d postgres minio litellm

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f postgres
docker-compose logs -f minio
docker-compose logs -f litellm
```

### 2. Database Initialization
Initialize database schema on first run:

```bash
# Apply schema
docker exec -i centrifuge_postgres psql -U centrifuge -d centrifuge < ops/sql/init/01_schema.sql

# Verify schema
docker exec -it centrifuge_postgres psql -U centrifuge -d centrifuge -c "\dt"
```

### 3. Start API Service
```bash
# Development mode
python api/main_v2.py

# Production mode with gunicorn
gunicorn api.main_v2:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or using docker
docker-compose up -d api
```

### 4. Start Worker Processes
```bash
# Start single worker
python worker/main.py

# Start multiple workers
for i in {1..4}; do
    WORKER_ID="worker-$i" python worker/main.py &
done

# Or using docker-compose (scales to 4 workers)
docker-compose up -d --scale worker=4 worker
```

### 5. Verify System Health
```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "database": "healthy",
  "storage": "healthy",
  "timestamp": "2025-01-13T17:34:28Z"
}
```

## Shutdown Procedures

### Graceful Shutdown
1. Stop accepting new runs:
   ```bash
   # Set API to maintenance mode (if implemented)
   curl -X POST http://localhost:8000/admin/maintenance
   ```

2. Wait for running jobs to complete:
   ```bash
   # Check for running jobs
   curl http://localhost:8000/runs?state=running
   ```

3. Stop workers gracefully:
   ```bash
   # Send SIGTERM to workers (they will finish current run)
   pkill -TERM -f "worker/main.py"
   
   # Or if using docker-compose
   docker-compose stop worker
   ```

4. Stop API service:
   ```bash
   # Send SIGTERM to API
   pkill -TERM -f "api/main_v2.py"
   
   # Or if using docker-compose
   docker-compose stop api
   ```

5. Stop infrastructure:
   ```bash
   docker-compose down
   ```

### Emergency Shutdown
```bash
# Force stop all services immediately
docker-compose down -v

# Kill all python processes
pkill -9 -f "centrifuge"
```

## Health Checks

### API Health Check
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed component status
curl http://localhost:8000/health | jq
```

### Database Health Check
```bash
# Check database connectivity
docker exec centrifuge_postgres pg_isready -U centrifuge

# Check active connections
docker exec centrifuge_postgres psql -U centrifuge -d centrifuge \
  -c "SELECT count(*) FROM pg_stat_activity WHERE datname='centrifuge';"

# Check run queue depth
docker exec centrifuge_postgres psql -U centrifuge -d centrifuge \
  -c "SELECT state, COUNT(*) FROM runs GROUP BY state;"
```

### Storage Health Check
```bash
# Check MinIO status
curl http://localhost:9000/minio/health/live
curl http://localhost:9000/minio/health/ready

# List buckets
docker exec centrifuge_minio mc ls local/
```

### Worker Health Check
```bash
# Check worker heartbeats (stale if > 5 minutes)
docker exec centrifuge_postgres psql -U centrifuge -d centrifuge -c "
  SELECT claimed_by, heartbeat_at, 
         NOW() - heartbeat_at as time_since_heartbeat
  FROM runs 
  WHERE state = 'running'
  ORDER BY heartbeat_at DESC;"
```

## Common Failure Modes

### 1. Database Connection Failures
**Symptoms:**
- API returns 503 with database="unhealthy"
- Workers fail to claim runs
- Errors: "Failed to connect to database"

**Resolution:**
```bash
# Check database status
docker-compose ps postgres
docker-compose logs postgres

# Restart database
docker-compose restart postgres

# If persistent issues, check disk space
df -h
docker system df
```

### 2. MinIO Storage Failures
**Symptoms:**
- API returns 503 with storage="unhealthy"
- Artifact downloads fail
- Errors: "Failed to upload/download from storage"

**Resolution:**
```bash
# Check MinIO status
docker-compose ps minio
docker-compose logs minio

# Restart MinIO
docker-compose restart minio

# Check MinIO console
open http://localhost:9001
# Default credentials: minioadmin/minioadmin
```

### 3. LLM API Failures
**Symptoms:**
- Runs fail during LLM processing phase
- Errors: "LLM processing failed", "Invalid API key", "Rate limit exceeded"

**Resolution:**
```bash
# Check LiteLLM proxy
docker-compose logs litellm

# Verify API key is set
echo $OPENAI_API_KEY

# Test LiteLLM directly
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Enable mock mode if API issues persist
export USE_MOCK_LLM=true
```

### 4. Worker Stalls
**Symptoms:**
- Runs stuck in "running" state
- No heartbeat updates
- Queue backing up

**Resolution:**
```bash
# Identify stale runs (no heartbeat > 5 minutes)
docker exec centrifuge_postgres psql -U centrifuge -d centrifuge -c "
  SELECT run_id, claimed_by, heartbeat_at
  FROM runs
  WHERE state = 'running'
    AND heartbeat_at < NOW() - INTERVAL '5 minutes';"

# Reset stale runs to queued
docker exec centrifuge_postgres psql -U centrifuge -d centrifuge -c "
  UPDATE runs
  SET state = 'queued', claimed_by = NULL
  WHERE state = 'running'
    AND heartbeat_at < NOW() - INTERVAL '5 minutes';"

# Restart workers
docker-compose restart worker
```

### 5. Row Limit Exceeded
**Symptoms:**
- Run fails immediately after submission
- Error: "Input file exceeds 50,000 row limit"

**Resolution:**
```bash
# Split large files before processing
split -l 50000 large_file.csv chunk_

# Or increase limit (requires code change)
# Edit core/pipeline.py line 88: if row_count > 50000:
```

## Configuration

### Environment Variables

#### Core Settings
```bash
# Database
POSTGRES_HOST=postgres        # Database host
POSTGRES_PORT=5432            # Database port
POSTGRES_DB=centrifuge        # Database name
POSTGRES_USER=centrifuge      # Database user
POSTGRES_PASSWORD=centrifuge  # Database password

# Storage
USE_LOCAL_STORAGE=false       # Use local filesystem instead of MinIO
MINIO_ENDPOINT=minio:9000     # MinIO endpoint
MINIO_ACCESS_KEY=minioadmin   # MinIO access key
MINIO_SECRET_KEY=minioadmin   # MinIO secret key
MINIO_BUCKET=centrifuge       # Storage bucket name

# LLM
USE_MOCK_LLM=false            # Use mock LLM for testing
LITELLM_URL=http://litellm:4000  # LiteLLM proxy URL
OPENAI_API_KEY=sk-...        # OpenAI API key (required for production)
```

#### Worker Settings
```bash
# Worker Configuration
HEARTBEAT_INTERVAL=30         # Seconds between heartbeats
VISIBILITY_TIMEOUT=300        # Seconds before run can be reclaimed
POLL_INTERVAL=5              # Seconds between queue polls
WORKER_ID=worker-1           # Unique worker identifier (auto-generated if not set)
```

### Configuration Files

#### Schema Configuration
Location: `schemas/v1.yaml`
- Defines column specifications
- Validation rules
- Canonical values

#### LLM Prompts
Location: `prompts/`
- `department.yaml`: Department canonicalization prompts
- `account_name.yaml`: Account name canonicalization prompts

## Monitoring

### Key Metrics to Monitor

#### System Metrics
- **Queue Depth**: Number of runs in 'queued' state
- **Processing Rate**: Runs completed per hour
- **Error Rate**: Failed runs / total runs
- **Worker Utilization**: Active workers / total workers

#### Performance Metrics
- **Run Duration**: Average time from queued to completed
- **Phase Duration**: Time spent in each processing phase
- **LLM Latency**: Average LLM response time
- **Cache Hit Rate**: Cached mappings / total mappings

### Monitoring Queries

```sql
-- Queue depth by state
SELECT state, COUNT(*) as count
FROM runs
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY state;

-- Average processing time (last hour)
SELECT AVG(completed_at - started_at) as avg_duration
FROM runs
WHERE state = 'succeeded'
  AND completed_at > NOW() - INTERVAL '1 hour';

-- Error rate
SELECT 
  COUNT(CASE WHEN state = 'failed' THEN 1 END)::float / 
  COUNT(*)::float as error_rate
FROM runs
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Cache hit rate
SELECT 
  SUM(use_count) as total_uses,
  COUNT(*) as unique_mappings
FROM canonical_mappings
WHERE last_used_at > NOW() - INTERVAL '1 hour';
```

### Alerting Thresholds
- Queue depth > 100: Check worker health
- Error rate > 10%: Check logs for systematic issues
- Worker heartbeat stale > 5 minutes: Restart worker
- Database connections > 90% of max: Scale connection pool
- Storage usage > 80%: Clean up old artifacts

## Troubleshooting

### Debug Mode
Enable detailed logging:
```bash
# Set log level
export LOG_LEVEL=DEBUG

# Enable SQL query logging
export POSTGRES_LOG_STATEMENT=all
```

### Common Issues

#### "No runs being processed"
1. Check workers are running: `ps aux | grep worker`
2. Check database connectivity: `curl http://localhost:8000/health`
3. Check for stale locks in database
4. Restart workers if needed

#### "LLM mappings incorrect"
1. Check prompt templates in `prompts/`
2. Verify canonical values in schema
3. Review cache for bad mappings
4. Clear cache if needed:
   ```sql
   DELETE FROM canonical_mappings 
   WHERE confidence < 0.5;
   ```

#### "Artifacts not accessible"
1. Check MinIO is running
2. Verify bucket exists
3. Check storage path in artifacts table
4. Regenerate artifacts if needed

### Log Locations
- API logs: `stdout` or `/var/log/centrifuge/api.log`
- Worker logs: `stdout` or `/var/log/centrifuge/worker-*.log`
- Database logs: `docker-compose logs postgres`
- MinIO logs: `docker-compose logs minio`
- LiteLLM logs: `docker-compose logs litellm`

## Emergency Procedures

### Data Recovery
```bash
# Backup database
docker exec centrifuge_postgres pg_dump -U centrifuge centrifuge > backup.sql

# Restore database
docker exec -i centrifuge_postgres psql -U centrifuge centrifuge < backup.sql

# Export artifacts from MinIO
docker exec centrifuge_minio mc mirror local/centrifuge /backup/
```

### Reset System
```bash
# Stop all services
docker-compose down

# Clear all data (WARNING: Destructive!)
docker volume rm centrifuge_postgres_data
docker volume rm centrifuge_minio_data

# Restart fresh
docker-compose up -d
```

### Manual Run Recovery
```sql
-- Reset failed run to queued
UPDATE runs 
SET state = 'queued', 
    claimed_by = NULL,
    error_message = NULL,
    error_code = NULL
WHERE run_id = 'UUID-HERE';

-- Mark run as failed manually
UPDATE runs
SET state = 'failed',
    completed_at = NOW(),
    error_message = 'Manual intervention',
    error_code = 'MANUAL_FAIL'
WHERE run_id = 'UUID-HERE';
```

## Performance Tuning

### Database Tuning
```sql
-- Increase connection pool
ALTER SYSTEM SET max_connections = 200;

-- Optimize for SSD
ALTER SYSTEM SET random_page_cost = 1.1;

-- Increase work memory for complex queries
ALTER SYSTEM SET work_mem = '256MB';

-- Reload configuration
SELECT pg_reload_conf();
```

### Worker Scaling
```bash
# Scale workers based on queue depth
QUEUE_DEPTH=$(curl -s http://localhost:8000/runs?state=queued | jq '.total')

if [ $QUEUE_DEPTH -gt 50 ]; then
  docker-compose up -d --scale worker=8 worker
elif [ $QUEUE_DEPTH -lt 10 ]; then
  docker-compose up -d --scale worker=2 worker
fi
```

### Cache Optimization
```sql
-- Prune old unused mappings
DELETE FROM canonical_mappings
WHERE last_used_at < NOW() - INTERVAL '30 days'
  AND use_count < 5;

-- Create partial index for hot mappings
CREATE INDEX idx_hot_mappings 
ON canonical_mappings(column_name, variant_value)
WHERE use_count > 10;
```

## Support

For issues or questions:
1. Check logs for error messages
2. Consult this runbook
3. Review code documentation
4. Contact the development team

Remember: Always backup data before making system changes!