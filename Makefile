.PHONY: help install test lint run docker-up docker-down pipeline

help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e ".[dev,tuning]"

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Run linters
	ruff check src/ tests/
	mypy src/

run: ## Run the API server
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

docker-up: ## Start all services
	docker-compose up -d --build

docker-down: ## Stop all services
	docker-compose down

pipeline: ## Run the demo pipeline
	python scripts/run_pipeline.py