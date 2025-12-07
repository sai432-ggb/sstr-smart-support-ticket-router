# Smart Support Ticket Router - Makefile
# Automation for common development tasks

.PHONY: help install setup data train test api docker clean

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Smart Support Ticket Router - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## Install all dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

setup: ## Set up project directories and environment
	@echo "$(BLUE)Setting up project...$(NC)"
	mkdir -p data/raw data/processed data/features
	mkdir -p models/saved_models models/metrics models/experiments
	mkdir -p logs tests
	mkdir -p notebooks
	cp .env.example .env || echo "API_TOKENS=test-token-123" > .env
	@echo "$(GREEN)✓ Project setup complete$(NC)"

data: ## Generate synthetic training data
	@echo "$(BLUE)Generating training data...$(NC)"
	python scripts/generate_data.py
	@echo "$(GREEN)✓ Training data generated$(NC)"

train: ## Train all ML models
	@echo "$(BLUE)Training models...$(NC)"
	python scripts/train_models.py
	@echo "$(GREEN)✓ Models trained$(NC)"

evaluate: ## Evaluate model performance
	@echo "$(BLUE)Evaluating models...$(NC)"
	python scripts/evaluate_models.py
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-quick: ## Run quick tests (no coverage)
	@echo "$(BLUE)Running quick tests...$(NC)"
	pytest tests/ -v -x
	@echo "$(GREEN)✓ Quick tests complete$(NC)"

api: ## Start API server (development)
	@echo "$(BLUE)Starting API server...$(NC)"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api-prod: ## Start API server (production)
	@echo "$(BLUE)Starting API server (production)...$(NC)"
	gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t sstr:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -p 8000:8000 -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models sstr:latest

docker-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "$(YELLOW)API available at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)API docs at: http://localhost:8000/docs$(NC)"

docker-down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-logs: ## View docker logs
	docker-compose logs -f api

docker-clean: ## Clean Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker-compose down -v
	docker system prune -f
	@echo "$(GREEN)✓ Docker resources cleaned$(NC)"

notebook: ## Start Jupyter notebook
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	jupyter notebook notebooks/

lint: ## Run code linting
	@echo "$(BLUE)Running linters...$(NC)"
	black src/ tests/ --check
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/ scripts/
	@echo "$(GREEN)✓ Code formatted$(NC)"

clean: ## Clean generated files
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	rm -rf build dist *.egg-info
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-data: ## Clean generated data (WARNING: deletes training data)
	@echo "$(RED)WARNING: This will delete all generated data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/raw/* data/processed/* data/features/*; \
		echo "$(GREEN)✓ Data cleaned$(NC)"; \
	fi

clean-models: ## Clean trained models (WARNING: deletes models)
	@echo "$(RED)WARNING: This will delete all trained models!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/saved_models/* models/metrics/* models/experiments/*; \
		echo "$(GREEN)✓ Models cleaned$(NC)"; \
	fi

all: setup install data train ## Set up everything from scratch
	@echo "$(GREEN)✓ Complete setup finished!$(NC)"
	@echo "$(YELLOW)Run 'make api' to start the API server$(NC)"

check: ## Run all checks (lint, test)
	@echo "$(BLUE)Running all checks...$(NC)"
	make lint
	make test
	@echo "$(GREEN)✓ All checks passed$(NC)"

status: ## Show project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo ""
	@echo "$(YELLOW)Data:$(NC)"
	@test -d data/raw && echo "  ✓ Raw data directory exists" || echo "  ✗ Raw data directory missing"
	@test -f data/raw/train_tickets.csv && echo "  ✓ Training data exists" || echo "  ✗ Training data missing"
	@echo ""
	@echo "$(YELLOW)Models:$(NC)"
	@test -f models/saved_models/classifier_v1.pkl && echo "  ✓ Classifier trained" || echo "  ✗ Classifier not trained"
	@test -f models/saved_models/forecaster_v1.pkl && echo "  ✓ Forecaster trained" || echo "  ✗ Forecaster not trained"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@docker images | grep sstr > /dev/null 2>&1 && echo "  ✓ Docker image exists" || echo "  ✗ Docker image not built"
	@docker-compose ps 2>/dev/null | grep Up > /dev/null 2>&1 && echo "  ✓ Services running" || echo "  ✗ Services not running"

dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	make install
	make setup
	make data
	make train
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo "$(YELLOW)Run 'make api' to start the server$(NC)"