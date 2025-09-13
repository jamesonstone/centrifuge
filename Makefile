# Centrifuge Makefile v2 - Complete operational targets
.PHONY: help setup test lint clean format check run
.PHONY: infra-start infra-stop infra-reset infra-logs
.PHONY: api-start api-start-prod api-logs
.PHONY: worker-start worker-scale worker-logs
.PHONY: stack-up stack-down stack-restart stack-logs stack-status
.PHONY: db-migrate db-backup db-restore db-console db-status
.PHONY: health-check queue-status worker-status cache-stats
.PHONY: test-unit test-integration test-concurrency test-all test-cov
.PHONY: demo demo-submit demo-status demo-clean

# Variables
WORKERS ?= 4
BACKUP_DIR := backups
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
API_URL := http://localhost:8080
MINIO_URL := http://localhost:9001

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║            Centrifuge - Data Cleaning Pipeline                    ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make stack-up         - Start complete stack"
	@echo "  make demo            - Run interactive demo"
	@echo "  make health-check    - Check system health"
	@echo ""
	@echo "📦 Infrastructure:"
	@echo "  make infra-start     - Start infrastructure (DB, MinIO, LiteLLM)"
	@echo "  make infra-stop      - Stop infrastructure"
	@echo "  make infra-reset     - Reset infrastructure (DESTRUCTIVE)"
	@echo "  make infra-logs      - Show infrastructure logs"
	@echo ""
	@echo "🔧 Services:"
	@echo "  make api-start       - Start API service (dev mode)"
	@echo "  make api-start-prod  - Start API service (production mode)"
	@echo "  make worker-start    - Start single worker"
	@echo "  make worker-scale    - Scale workers (WORKERS=n)"
	@echo ""
	@echo "🎯 Stack Management:"
	@echo "  make stack-up        - Start complete stack with docker-compose"
	@echo "  make stack-down      - Stop complete stack"
	@echo "  make stack-restart   - Restart complete stack"
	@echo "  make stack-logs      - Show all logs"
	@echo "  make stack-status    - Show stack status"
	@echo ""
	@echo "🗄️ Database:"
	@echo "  make db-migrate      - Apply database migrations"
	@echo "  make db-backup       - Backup database"
	@echo "  make db-restore FILE=<file> - Restore database"
	@echo "  make db-console      - Open database console"
	@echo "  make db-status       - Show database status"
	@echo ""
	@echo "📊 Monitoring:"
	@echo "  make health-check    - System health check"
	@echo "  make queue-status    - Show queue status"
	@echo "  make worker-status   - Show worker status"
	@echo "  make cache-stats     - Show cache statistics"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test-unit       - Run unit tests"
	@echo "  make test-integration- Run integration tests"
	@echo "  make test-concurrency- Run concurrency tests"
	@echo "  make test-all        - Run all tests"
	@echo "  make test-cov        - Run tests with coverage"
	@echo ""
	@echo "🛠️ Development:"
	@echo "  make setup           - Install dependencies"
	@echo "  make lint            - Run linters"
	@echo "  make format          - Format code"
	@echo "  make clean           - Clean build artifacts"
	@echo "  make check           - Run all checks"

# ============================================================================
# Setup and Dependencies
# ============================================================================

setup:
	@echo "📦 Installing dependencies..."
	uv sync --extra dev
	uv pip install -e .
	@echo "✅ Dependencies installed"

run: api-start

clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "✅ Clean complete"

# ============================================================================
# Infrastructure Management
# ============================================================================

infra-start:
	@echo "🚀 Starting infrastructure services..."
	docker-compose up -d postgres minio minio-setup litellm
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@$(MAKE) db-migrate
	@echo "✅ Infrastructure ready"

infra-stop:
	@echo "🛑 Stopping infrastructure services..."
	docker-compose stop postgres minio litellm
	@echo "✅ Infrastructure stopped"

infra-reset:
	@echo "⚠️  WARNING: This will delete all data!"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	@echo "🔄 Resetting infrastructure..."
	docker-compose down -v
	docker-compose up -d postgres minio minio-setup litellm
	@sleep 10
	@$(MAKE) db-migrate
	@echo "✅ Infrastructure reset complete"

infra-logs:
	docker-compose logs -f postgres minio litellm

# ============================================================================
# Application Services
# ============================================================================

api-start:
	@echo "🌐 Starting API service (dev mode)..."
	@if [ -f .env ]; then set -a && source .env && set +a; fi && \
	POSTGRES_HOST=localhost POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres POSTGRES_DB=centrifuge \
	ARTIFACT_ENDPOINT=http://localhost:9000 ARTIFACT_ACCESS_KEY=minioadmin ARTIFACT_SECRET_KEY=minioadmin \
	MINIO_ENDPOINT=localhost:9000 \
	OPENAI_API_KEY="$${OPENAI_API_KEY}" \
	USE_LOCAL_STORAGE=false uv run python api/main.py

api-start-prod:
	@echo "🌐 Starting API service (production mode)..."
	gunicorn api.main:app \
		-w 4 \
		-k uvicorn.workers.UvicornWorker \
		--bind 0.0.0.0:8000 \
		--access-logfile - \
		--error-logfile -

api-logs:
	docker-compose logs -f centrifuge-app

worker-start:
	@echo "⚙️  Starting worker..."
	@if [ -f .env ]; then set -a && source .env && set +a; fi && \
	POSTGRES_HOST=localhost POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres POSTGRES_DB=centrifuge \
	ARTIFACT_ENDPOINT=http://localhost:9000 ARTIFACT_ACCESS_KEY=minioadmin ARTIFACT_SECRET_KEY=minioadmin \
	MINIO_ENDPOINT=localhost:9000 \
	OPENAI_API_KEY="$${OPENAI_API_KEY}" \
	USE_LOCAL_STORAGE=false uv run python worker/main.py

worker-scale:
	@echo "📈 Scaling workers to $(WORKERS) instances..."
	@for i in $$(seq 1 $(WORKERS)); do \
		echo "Starting worker-$$i..."; \
		WORKER_ID="worker-$$i" USE_LOCAL_STORAGE=false uv run python worker/main.py & \
	done
	@echo "✅ Started $(WORKERS) workers"

worker-logs:
	docker-compose logs -f centrifuge-worker-1 centrifuge-worker-2

# ============================================================================
# Stack Management
# ============================================================================

stack-up:
	@echo "🚀 Starting complete stack..."
	docker-compose up -d
	@echo "⏳ Waiting for services..."
	@sleep 15
	@$(MAKE) health-check
	@echo ""
	@echo "✅ Stack is ready!"
	@echo "📍 Endpoints:"
	@echo "   API:     $(API_URL)"
	@echo "   MinIO:   $(MINIO_URL) (minioadmin/minioadmin)"
	@echo "   LiteLLM: http://localhost:4000"

stack-down:
	@echo "🛑 Stopping complete stack..."
	docker-compose down
	@echo "✅ Stack stopped"

stack-restart: stack-down stack-up

stack-logs:
	docker-compose logs -f

stack-status:
	@echo "📊 Stack Status:"
	@echo "=================="
	@docker-compose ps
	@echo ""
	@$(MAKE) health-check

# ============================================================================
# Database Operations
# ============================================================================

db-migrate:
	@echo "🔄 Applying database migrations..."
	@docker exec -i centrifuge-postgres psql -U postgres -d centrifuge < ops/sql/init/01_schema.sql 2>/dev/null || \
		echo "Schema already exists or database not ready"
	@echo "✅ Migrations complete"

db-backup:
	@echo "💾 Backing up database..."
	@mkdir -p $(BACKUP_DIR)
	docker exec centrifuge-postgres pg_dump -U postgres centrifuge > $(BACKUP_DIR)/centrifuge_$(TIMESTAMP).sql
	@echo "✅ Backup saved to $(BACKUP_DIR)/centrifuge_$(TIMESTAMP).sql"

db-restore:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Error: Please specify FILE=<backup_file>"; \
		exit 1; \
	fi
	@echo "📥 Restoring database from $(FILE)..."
	docker exec -i centrifuge-postgres psql -U postgres centrifuge < $(FILE)
	@echo "✅ Database restored"

db-console:
	@echo "🖥️  Opening database console..."
	docker exec -it centrifuge-postgres psql -U postgres -d centrifuge

db-status:
	@echo "📊 Database Status:"
	@docker exec centrifuge-postgres psql -U postgres -d centrifuge -c \
		"SELECT version();" 2>/dev/null || echo "Database not accessible"
	@echo ""
	@echo "Active connections:"
	@docker exec centrifuge-postgres psql -U postgres -d centrifuge -c \
		"SELECT count(*) as connections FROM pg_stat_activity WHERE datname='centrifuge';" 2>/dev/null || true

# ============================================================================
# Monitoring and Health
# ============================================================================

health-check:
	@echo "🏥 System Health Check:"
	@echo "======================="
	@curl -s $(API_URL)/health 2>/dev/null | jq '.' || echo "❌ API not responding"
	@echo ""
	@$(MAKE) queue-status

queue-status:
	@echo "📊 Queue Status:"
	@docker exec centrifuge-postgres psql -U postgres -d centrifuge -t -c \
		"SELECT state, COUNT(*) as count FROM runs WHERE created_at > NOW() - INTERVAL '1 hour' GROUP BY state ORDER BY state;" 2>/dev/null || \
		echo "Database not accessible"

worker-status:
	@echo "⚙️  Worker Status:"
	@docker exec centrifuge-postgres psql -U postgres -d centrifuge -c \
		"SELECT claimed_by, heartbeat_at, NOW() - heartbeat_at as time_since_heartbeat \
		 FROM runs WHERE state = 'running' ORDER BY heartbeat_at DESC;" 2>/dev/null || \
		echo "No active workers or database not accessible"

cache-stats:
	@echo "💾 Cache Statistics:"
	@docker exec centrifuge-postgres psql -U postgres -d centrifuge -c \
		"SELECT COUNT(*) as total_mappings, \
		        COUNT(DISTINCT column_name) as unique_columns, \
		        COUNT(DISTINCT variant_value) as unique_variants, \
		        COALESCE(SUM(use_count), 0) as total_uses, \
		        COALESCE(AVG(confidence)::numeric(3,2), 0) as avg_confidence \
		 FROM canonical_mappings WHERE superseded_at IS NULL;" 2>/dev/null || \
		echo "Cache not accessible"

# ============================================================================
# Testing
# ============================================================================

test-unit:
	@echo "🧪 Running unit tests..."
	uv run python -m pytest tests/test_determinism.py tests/test_golden.py tests/test_adversarial.py -v

test-integration:
	@echo "🧪 Running integration tests..."
	uv run python -m pytest tests/test_integration.py -v

test-concurrency:
	@echo "🧪 Running concurrency tests..."
	uv run python -m pytest tests/test_worker_concurrency.py -v --asyncio-mode=auto

test-all:
	@echo "🧪 Running all tests..."
	uv run python -m pytest tests/ -v --asyncio-mode=auto

test-cov:
	@echo "📊 Running tests with coverage..."
	uv run python -m pytest tests/ --cov=core --cov=api --cov=worker \
		--cov-report=term-missing \
		--cov-report=html \
		--asyncio-mode=auto
	@echo "Coverage report saved to htmlcov/index.html"

# ============================================================================
# Code Quality
# ============================================================================

lint:
	@echo "🔍 Running linters..."
	uv run flake8 core/ api/ worker/ --max-line-length=120
	uv run pylint core/ api/ worker/ --disable=C0114,C0115,C0116
	uv run mypy core/ api/ worker/ --ignore-missing-imports

format:
	@echo "✨ Formatting code..."
	uv run black core/ api/ worker/ tests/
	uv run isort core/ api/ worker/ tests/

check: lint test-all
	@echo "✅ All checks passed!"

# ============================================================================
# Demo and Examples
# ============================================================================

demo: stack-up
	@echo ""
	@echo "🎮 Interactive Demo"
	@echo "=================="
	@echo ""
	@echo "1️⃣  Submit a test run:"
	@echo "   curl -X POST $(API_URL)/runs \\"
	@echo "     -F 'file=@sample-data/sample_data.csv' \\"
	@echo "     -F 'schema_version=v1' \\"
	@echo "     -F 'llm_columns=Department,Account Name'"
	@echo ""
	@echo "2️⃣  Check run status:"
	@echo "   curl $(API_URL)/runs/<run_id>"
	@echo ""
	@echo "3️⃣  List artifacts:"
	@echo "   curl $(API_URL)/runs/<run_id>/artifacts"
	@echo ""
	@echo "4️⃣  Download cleaned data:"
	@echo "   curl $(API_URL)/runs/<run_id>/artifacts/cleaned/download -o cleaned.csv"
	@echo ""
	@echo "Press any key to submit a test run..."
	@read -n 1
	@$(MAKE) demo-submit

demo-submit:
	@echo "📤 Submitting test run..."
	@RUN_ID=$$(curl -s -X POST $(API_URL)/runs \
		-F "file=@sample-data/sample_data.csv" \
		-F "schema_version=v1" \
		-F "dry_run=false" \
		-F "llm_columns=Department,Account Name" | jq -r '.run_id'); \
	echo "✅ Run submitted: $$RUN_ID"; \
	echo ""; \
	echo "Checking status..."; \
	sleep 5; \
	curl -s $(API_URL)/runs/$$RUN_ID | jq

demo-status:
	@echo "📊 Recent runs:"
	@curl -s $(API_URL)/runs?limit=5 | jq

demo-clean:
	@echo "🧹 Cleaning demo data..."
	@docker exec centrifuge-postgres psql -U postgres -d centrifuge -c \
		"DELETE FROM runs WHERE schema_version = 'v1';"
	@echo "✅ Demo data cleaned"

# ============================================================================
# Troubleshooting
# ============================================================================

debug-env:
	@echo "🔍 Environment Variables:"
	@echo "POSTGRES_HOST: $${POSTGRES_HOST:-localhost}"
	@echo "POSTGRES_PORT: $${POSTGRES_PORT:-5432}"
	@echo "MINIO_ENDPOINT: $${MINIO_ENDPOINT:-minio:9000}"
	@echo "LITELLM_URL: $${LITELLM_URL:-http://litellm:4000}"
	@echo "USE_LOCAL_STORAGE: $${USE_LOCAL_STORAGE:-false}"
	@echo "USE_MOCK_LLM: $${USE_MOCK_LLM:-false}"
	@echo "OPENAI_API_KEY: $${OPENAI_API_KEY:+[SET]}"

reset-stale-runs:
	@echo "🔄 Resetting stale runs..."
	@docker exec centrifuge-postgres psql -U postgres -d centrifuge -c \
		"UPDATE runs SET state = 'queued', claimed_by = NULL \
		 WHERE state = 'running' AND heartbeat_at < NOW() - INTERVAL '5 minutes';"
	@echo "✅ Stale runs reset"

.DEFAULT_GOAL := help
