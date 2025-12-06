.PHONY: help install train evaluate deploy clean test

help:
	@echo "MLOps Makefile Commands:"
	@echo "  make set_env - Set up the Python environment"
	@echo "  make train     - Train the model"
	@echo "  make evaluate  - Evaluate model performance"
	@echo "  make test      - Run tests"
	@echo "  make deploy    - Deploy model"
	@echo "  make clean     - Clean artifacts"

set_env:
	source .venv/bin/activate

train:
	python src/train.py

evaluate:
	python src/evaluate.py

test:
	pytest tests/

deploy:
	python src/deploy.py

# clean:
# 	rm -rf __pycache__ .pytest_cache models/*.pkl data/processed/*
# 	find . -type d -name "__pycache__" -exec rm -rf {} +