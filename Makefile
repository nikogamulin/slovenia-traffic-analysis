.PHONY: help install install-dev test lint format clean run-notebook setup

help:
	@echo "Available commands:"
	@echo "  install       Install project dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run tests with coverage"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean build artifacts and cache"
	@echo "  setup         Complete project setup"
	@echo "  run-notebook  Start Jupyter Lab"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports
	pylint src/ --max-line-length=100

format:
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile=black --line-length=100

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

setup: install-dev
	@echo "Creating data directories..."
	mkdir -p data/raw data/processed data/external results/figures results/models results/reports logs cache
	@echo "Setup complete!"

run-notebook:
	jupyter lab --port=8888 --no-browser