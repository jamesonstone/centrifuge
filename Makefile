.PHONY: help setup run test lint clean migrate compose-up compose-down compose-logs compose-restart

# default target
help:
	@echo "Available targets:"
	@echo "  setup           - Install dependencies with uv"
	@echo "  run             - Run the API server locally"
	@echo "  worker          - Run a worker locally"
	@echo "  test            - Run tests"
	@echo "  lint            - Run linting and formatting"
	@echo "  clean           - Remove generated files and caches"
	@echo "  migrate         - Run database migrations"
	@echo "  compose-up      - Start all services with docker-compose"
	@echo "  compose-down    - Stop all services"
	@echo "  compose-logs    - Show docker-compose logs"
	@echo "  compose-restart - Restart all services"

setup:
	@echo "Installing dependencies with uv..."
	uv pip install -e ".[dev]"
	@echo "Dependencies installed!"

run:
	@echo "Starting API server..."
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

worker:
	@echo "Starting worker..."
	python worker/main.py

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=app --cov=worker --cov=core --cov-report=term-missing

lint:
	@echo "Running linters..."
	ruff check .
	black --check .
	mypy app/ worker/ core/

format:
	@echo "Formatting code..."
	black .
	ruff check --fix .

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	migrate:
	@echo "Running database migrations..."
	@echo "Note: In production, we would use Alembic. For PoC, migrations run automatically via docker-entrypoint."

compose-up:
	@echo "Starting docker-compose services..."
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@echo "Services started! API available at http://localhost:8080"

compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down

compose-logs:
	docker-compose logs -f

compose-restart: compose-down compose-up