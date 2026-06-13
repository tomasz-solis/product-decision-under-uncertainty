# Common commands for the Product Decision Under Uncertainty project.
# This repo is uv-first; every target shells through `uv run` so the pinned
# environment in uv.lock is always used.

.DEFAULT_GOAL := help
.PHONY: help install app artifacts evidence test lint format type check ci clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync the dev environment from uv.lock.
	uv sync --extra dev

app: ## Run the Streamlit exploration app.
	uv run streamlit run app.py

artifacts: ## Regenerate the published case-study artifacts.
	uv run python scripts/generate_case_study_artifacts.py

evidence: ## Refresh the public-evidence profile and parameter candidates.
	uv run python scripts/profile_public_evidence.py
	uv run python scripts/build_parameter_candidates.py

test: ## Run the test suite with coverage.
	uv run pytest -q

lint: ## Check lint rules with ruff.
	uv run ruff check .

format: ## Auto-format the codebase with ruff.
	uv run ruff format .

type: ## Run static type checks with mypy.
	uv run mypy app.py simulator tests

check: lint type test ## Run lint, type, and test gates.

ci: check artifacts ## Mirror CI: gates plus artifact freshness.
	git diff --exit-code -- CASE_STUDY.md EXECUTIVE_SUMMARY.md artifacts/case_study

clean: ## Remove caches and build leftovers.
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis *.egg-info
