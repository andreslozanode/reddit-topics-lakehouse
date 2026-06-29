.PHONY: help install lint format typecheck test sample validate deploy clean

ENV ?= dev

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Editable install with all extras
	pip install -e ".[spark,ingestion,ml,dev]"

lint:  ## Ruff lint + black format check
	ruff check src tests
	black --check src tests

format:  ## Auto-format with black + ruff --fix
	black src tests
	ruff check --fix src tests

typecheck:  ## Mypy static type-check
	mypy src

test:  ## Run unit + integration tests
	LAKEHOUSE_ENV=$(ENV) pytest -q

sample:  ## Generate synthetic SocialGrep-schema data
	python scripts/generate_sample_data.py --out .data/sample --topic demo --rows 2000

validate:  ## Validate the Databricks bundle
	databricks bundle validate -t $(ENV)

deploy:  ## Deploy the Databricks bundle
	databricks bundle deploy -t $(ENV)

clean:  ## Remove caches and build artifacts
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache \
	  spark-warehouse metastore_db derby.log src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
