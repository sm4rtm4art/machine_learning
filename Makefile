.PHONY: help install install-dev lint format typecheck test test-cov clean mlflow docs

# Default target
help:
	@echo "ML Portfolio - Available Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install all dependencies including dev tools"
	@echo "  make install-all   Install all optional dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  make lint          Run linter (ruff)"
	@echo "  make format        Format code (ruff format)"
	@echo "  make typecheck     Run type checker (mypy)"
	@echo "  make check         Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make mlflow        Start MLflow server (Docker)"
	@echo "  make mlflow-stop   Stop MLflow server"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove build artifacts and caches"
	@echo "  make pre-commit    Run pre-commit on all files"

# =============================================================================
# Setup
# =============================================================================

install:
	uv sync

install-dev:
	uv sync --all-extras

install-all:
	uv sync --all-extras

# =============================================================================
# Code Quality
# =============================================================================

lint:
	uv run ruff check src/ projects/

format:
	uv run ruff format src/ projects/
	uv run ruff check --fix src/ projects/

typecheck:
	uv run mypy src/

check: lint typecheck
	@echo "All checks passed!"

# =============================================================================
# Testing
# =============================================================================

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src/ml_portfolio --cov-report=html --cov-report=term-missing

test-smoke:
	uv run pytest -m smoke

# =============================================================================
# Infrastructure
# =============================================================================

mlflow:
	cd infra/mlflow && docker compose up -d
	@echo "MLflow UI available at http://localhost:5000"

mlflow-stop:
	cd infra/mlflow && docker compose down

mlflow-logs:
	cd infra/mlflow && docker compose logs -f

evidently:
	cd infra/monitoring/evidently && docker compose up -d
	@echo "Evidently UI available at http://localhost:8000"

evidently-stop:
	cd infra/monitoring/evidently && docker compose down

# =============================================================================
# Project Commands
# =============================================================================

# Run a project script: make run PROJECT=ocr_pipeline SCRIPT=train
run:
	uv run python projects/$(PROJECT)/scripts/$(SCRIPT).py

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/

pre-commit:
	uv run pre-commit run --all-files

pre-commit-install:
	uv run pre-commit install

# =============================================================================
# Documentation
# =============================================================================

docs:
	@echo "Documentation is in docs/ directory"
	@echo "View: docs/index.md"
