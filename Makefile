PYTHON := python

install:
	uv sync

format:
	black --line-length 100 src/ tests/

format-check:
	black --line-length 100 --check src/ tests/

test:
	pytest -v tests/