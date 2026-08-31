.PHONY: lint type test fmt

lint: ## Run ruff linter
	uv run ruff check .

type: ## Run mypy (strict) type checking
	uv run mypy src

test: ## Run test suite
	uv run pytest

fmt: ## Auto-format code and fix lint issues
	uv run ruff check . --fix
	uv run ruff format .
