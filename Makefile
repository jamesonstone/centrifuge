.PHONY: help setup run test lint clean migrate compose-up compose-down compose-logs compose-restart
.PHONY: test-golden test-determinism test-adversarial test-cov format check demo

# Default target
help:
	@echo "Centrifuge - Data Cleaning Pipeline"
	@echo ""
	@echo "Core targets:"
	@echo "  setup           - Install dependencies and prepare environment"
	@echo "  run             - Run the API server locally"
	@echo "  worker          - Run a worker locally"
	@echo "  test            - Run all tests"
	@echo "  lint            - Run linting and type checking"
	@echo "  format          - Format code with black and isort"
	@echo "  clean           - Remove generated files and caches"
	@echo "  check           - Run all checks (lint, type, test)"
	@echo ""
	@echo "Docker targets:"
	@echo "  compose-up      - Start all services with docker-compose"
	@echo "  compose-down    - Stop all services"
	@echo "  compose-logs    - Show docker-compose logs"
	@echo "  compose-restart - Restart all services"
	@echo "  migrate         - Run database migrations"
	@echo ""
	@echo "Test targets:"
	@echo "  test-golden     - Run golden tests"
	@echo "  test-determinism- Run determinism tests"
	@echo "  test-adversarial- Run adversarial tests"
	@echo "  test-cov        - Run tests with coverage report"
	@echo ""
	@echo "Quick start:"
	@echo "  demo            - Run demo (setup + compose + API)"

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

# Test targets
test-golden:
	@echo "Running golden tests..."
	pytest tests/test_golden.py -v

test-determinism:
	@echo "Running determinism tests..."
	pytest tests/test_determinism.py -v

test-adversarial:
	@echo "Running adversarial tests..."
	pytest tests/test_adversarial.py -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=core --cov=api --cov-report=term-missing --cov-report=html

# Run all checks
check: lint test
	@echo "All checks passed!"

# Quick demo
demo: compose-up setup
	@echo "Starting demo..."
	@sleep 5
	$(MAKE) run
