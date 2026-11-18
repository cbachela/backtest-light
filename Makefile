PYTHON := python

install:
	uv sync

install-editable:
	uv pip install -e .

format:
	black --line-length 100 src/ tests/

format-check:
	black --line-length 100 --check src/ tests/

test:
	pytest -v tests/

# All five models in parallel on the monthly panel with the CLI defaults
# (expanding/60/12 windows, first test 2006-01-01, 100 epochs).
MODELS = equal_weight gaussian gaussian_pto baseline baseline_pto
TRAIN = uv run --no-sync python demo/ml/train.py --panel data/monthly.npz

experiments:
	$(MAKE) -j5 $(MODELS)

equal_weight gaussian baseline:
	$(TRAIN) --model $@

gaussian_pto baseline_pto:
	$(TRAIN) --model $(@:_pto=) --no-e2e

.PHONY: install install-editable format format-check test experiments $(MODELS)
