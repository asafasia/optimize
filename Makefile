.PHONY: test lint format check-secrets clean-artifacts

test:
	python3 -m pytest

lint:
	python3 -m ruff check .

format:
	python3 -m ruff format .

check-secrets:
	gitleaks detect --no-git --redact --source .

clean-artifacts:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '.DS_Store' -o -name '*.log' -o -name 'laboneq_results.json' \) -delete
