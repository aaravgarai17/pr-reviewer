# Common tasks. `make help` lists them.
.DEFAULT_GOAL := help
.PHONY: help install test cov verify clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv and install dependencies
	python3 -m venv .venv
	./.venv/bin/pip install -q -r requirements.txt
	@echo "done — activate with: source .venv/bin/activate"

test:  ## Run the test suite
	pytest -q

cov:  ## Run tests with a coverage report
	pytest --cov=app --cov-report=term-missing -q

verify:  ## Run the full verification script
	./verify.sh

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage .coverage.* htmlcov

.PHONY: demo run ceiling
demo:  ## Run the full review pipeline offline, no API keys needed
	python demo.py

run:  ## Start the webhook server
	uvicorn app.main:app --reload

ceiling:  ## Measure the largest PR this bot can review completely
	python -m bench.diff_ceiling
