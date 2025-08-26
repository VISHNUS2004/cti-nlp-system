# Makefile for CTI-NLP System
# Use this for common development and deployment tasks

.PHONY: help setup install clean test lint format docker-build docker-up docker-down deploy

# Default target
help:
	@echo "CTI-NLP System - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup           - Run initial setup"
	@echo "  install         - Install Python dependencies"
	@echo "  clean           - Clean temporary files"
	@echo ""
	@echo "Development:"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint            - Run code linting"
	@echo "  format          - Format code with black and isort"
	@echo "  dev             - Start development server"
	@echo ""
	@echo "Docker Operations:"
	@echo "  docker-build    - Build Docker images"
	@echo "  docker-up       - Start all services with Docker Compose"
	@echo "  docker-down     - Stop all services"
	@echo "  docker-logs     - View service logs"
	@echo "  docker-clean    - Clean Docker resources"
	@echo ""
	@echo "ML Models:"
	@echo "  train-models    - Train all ML models"
	@echo "  train-classifier - Train threat classifier"
	@echo "  train-severity  - Train severity predictor"
	@echo ""
	@echo "Database:"
	@echo "  db-init         - Initialize database"
	@echo "  db-migrate      - Run database migrations"
	@echo "  db-seed         - Seed database with sample data"
	@echo "  db-backup       - Backup database"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-dev      - Deploy to development environment"
	@echo "  deploy-prod     - Deploy to production environment"

# Setup and Installation
setup:
	python setup.py

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

# Development
test:
	python tests/run_tests.py

test-unit:
	pytest tests/ -m "not integration" -v

test-integration:
	pytest tests/ -m integration -v

lint:
	flake8 backend/ database/ tests/
	black --check backend/ database/ tests/
	isort --check-only backend/ database/ tests/

format:
	black backend/ database/ tests/
	isort backend/ database/ tests/

dev:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Docker Operations
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker system prune -f

# ML Models
train-models: train-classifier train-severity

train-classifier:
	python scripts/train_threat_classifier.py

train-severity:
	python scripts/train_severity_model.py

# Database
db-init:
	python -c "from database.database import get_database_config; get_database_config().create_tables()"

db-migrate:
	alembic upgrade head

db-seed:
	python scripts/seed_database.py

db-backup:
	docker-compose exec postgres pg_dump -U cti_user cti_nlp > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Deployment
deploy-dev:
	docker-compose -f docker-compose.yml up -d

deploy-prod:
	@echo "Deploying to production..."
	git pull origin main
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "Production deployment complete!"

# Health checks
health:
	curl -f http://localhost:8000/health || echo "Service is not healthy"

status:
	docker-compose ps

# Monitoring
logs-app:
	docker-compose logs -f app

logs-db:
	docker-compose logs -f postgres

logs-redis:
	docker-compose logs -f redis

# Security
security-scan:
	pip-audit
	safety check

# Documentation
docs-serve:
	cd docs && python -m http.server 8080

# Performance testing
load-test:
	@echo "Running load tests..."
	# Add load testing commands here (e.g., with locust or wrk)

# Backup and restore
backup-full:
	@echo "Creating full backup..."
	mkdir -p backups/$(shell date +%Y%m%d)
	docker-compose exec postgres pg_dump -U cti_user cti_nlp > backups/$(shell date +%Y%m%d)/database.sql
	cp -r models/ backups/$(shell date +%Y%m%d)/
	cp -r data/ backups/$(shell date +%Y%m%d)/

restore:
	@echo "Restore requires manual intervention. Check backups/ directory."

# Environment management
env-dev:
	cp .env.example .env
	@echo "Development environment file created. Please update .env with your settings."

env-prod:
	@echo "Production environment setup requires manual configuration."
	@echo "Please ensure all production secrets are properly configured."

# Quick start
quick-start: setup install train-models docker-up
	@echo "Quick start complete! Visit http://localhost:8000/docs"

# Full reset (destructive!)
reset-all: docker-down clean
	docker volume prune -f
	rm -rf data/*
	rm -rf logs/*
	rm -rf models/*.pkl
	@echo "System reset complete. Run 'make quick-start' to rebuild."
