.PHONY: help setup-uv check-uv install install-dev install-all quickstart lint format typecheck test test-cov test-smoke clean mlflow mlflow-stop mlflow-logs  mlflow-pg mlflow-pg-stop mlflow-pg-logs evidently evidently-stop run pre-commit pre-commit-install docs

# Detect OS for platform-specific commands
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)

# Default target
help:
	@echo "ML Portfolio - Available Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup-uv      Install uv package manager (auto-detects OS)"
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
	@echo "  make mlflow            Start MLflow server (Docker)"
	@echo "  make mlflow-stop       Stop MLflow server"
	@echo "  make mlflow-logs       Show MLflow logs"
	@echo ""
	@echo "  make mlflow-pg         Start MLflow server with PostgreSQL (Docker)"
	@echo "  make mlflow-pg-stop    Stop MLflow (PostgreSQL) server"
	@echo "  make mlflow-pg-logs    Show MLflow (PostgreSQL) logs "
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove build artifacts and caches"
	@echo "  make pre-commit    Run pre-commit on all files"

# =============================================================================
# Setup
# =============================================================================

# Install uv package manager (auto-detects OS: Linux, macOS, Windows)
setup-uv:
	@echo "Detecting OS: $(UNAME_S)"
ifeq ($(UNAME_S),Linux)
	@echo "Installing uv for Linux..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo ""
	@echo "Add to your shell profile: export PATH=\"\$$HOME/.local/bin:\$$PATH\""
else ifeq ($(UNAME_S),Darwin)
	@echo "Installing uv for macOS..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo ""
	@echo "Add to your shell profile: export PATH=\"\$$HOME/.local/bin:\$$PATH\""
else
	@echo "For Windows, run in PowerShell:"
	@echo "  irm https://astral.sh/uv/install.ps1 | iex"
	@echo ""
	@echo "Or install via pipx/pip: pipx install uv"
endif
	@echo ""
	@echo "Then restart your terminal and run: make install-dev"

# Check if uv is installed, provide helpful message if not
check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Error: uv is not installed or not in PATH"; \
		echo "Run 'make setup-uv' to install it"; \
		exit 1; \
	}

install: check-uv
	uv sync

install-dev: check-uv
	uv sync --all-extras

install-all: check-uv
	uv sync --all-extras

# Quick start for new collaborators
quickstart: setup-uv
	@echo ""
	@echo "=== Next Steps ==="
	@echo "1. Restart your terminal (or run: source ~/.bashrc / source ~/.zshrc)"
	@echo "2. Run: make install-dev"
	@echo "3. Run: make pre-commit-install"
	@echo "4. You're ready! Try: make test"

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

mlflow-pg:
	cd infra/mlflow_pg && docker compose up -d
	@echo "MLflow UI available at http://localhost:5000"

mlflow-pg-stop:
	cd infra/mlflow_pg && docker compose down

mlflow-pg-logs:
	cd infra/mlflow_pg && docker compose logs -f

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
