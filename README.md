# btlight

A lightweight library for equity portfolio optimization and backtesting, with
point and Gaussian return models trained through a differentiable portfolio layer.

## Quick start (uv)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these
commands from the repository root. The portfolio experiments require Python 3.10
or newer; the commands below use Python 3.11.

```sh
uv sync --python 3.11
uv run --no-sync python demo/ml/train.py --smoke --epochs 1
```

This runs a short experiment on synthetic data. Results and a timestamped training
log are saved in a dated run folder under `outputs/`.

## Main files

| File | Purpose |
| --- | --- |
| `demo/ml/train.py` | Experiment entry point and command-line options |
| `src/btlight/ml/training/e2e.py` | Walk-forward training and held-out evaluation |
| `src/btlight/ml/model/return_predictor.py` | Point and Gaussian return networks |
| `src/btlight/ml/model/ensemble.py` | Gaussian mixture scoring and ensemble moments |
| `src/btlight/optimization/differentiable.py` | Differentiable portfolio allocation and loss |
| `src/btlight/ml/training/portfolio_data.py` | Splits, risk estimates, holdings alignment, and results |
| `src/btlight/data/daily.py` | Daily features, targets, and panel preparation |
| `demo/ml/plot_loss.py` | Training and validation loss plots |
| `demo/ml/thesis_tutorial.ipynb` | Simulated examples of prediction and E2E learning |

## Prepare data

For market/JKP data, provide a directory containing `market_data.parquet` and
`df_jkp.parquet` in the format read by `read_sources()` in
`src/btlight/data/daily.py`. Datasets are not included in the repository.

```sh
uv run --no-sync python demo/data/prepare_daily.py \
  --source /path/to/market-and-jkp \
  --start 2020-01-01 --end 2024-12-31 \
  --output data/daily.npz
```

Preparation saves the panel, settings, and audits, including `data/daily.exits.parquet`.
Use `--overwrite` to rebuild. Reviewed adjustments can be supplied with
`--split-corrections /path/to/corrections.parquet` and
`--terminal-events /path/to/corporate_actions.parquet`.

## Run experiments

The models use quarterly walk-forward windows: 39 months for training, nine for
validation, and three for testing. Learned portfolios are long-only, fully invested,
and capped at 10% per asset. All optimized portfolios use variance risk,
with objective `-mu @ w + risk_aversion * (w.T @ covariance @ w)` plus the
optional turnover penalty.

Gaussian E2E predicts return means and variances. The point baseline predicts means
and uses historical risk estimates. Both pretrain on forecast loss, then fine-tune
through allocation. Gaussian E2E also retains an NLL term during fine-tuning.

```sh
# Gaussian E2E
uv run --no-sync python demo/ml/train.py --model gaussian \
  --panel data/daily.npz --exit-audit data/daily.exits.parquet \
  --first-test 2024-01-01 --epochs 100

# Point E2E
uv run --no-sync python demo/ml/train.py --model baseline \
  --panel data/daily.npz --exit-audit data/daily.exits.parquet \
  --first-test 2024-01-01 --epochs 100

# Gaussian predict-then-optimize (PTO): NLL training followed by allocation
uv run --no-sync python demo/ml/train.py --model gaussian --no-e2e \
  --panel data/daily.npz --exit-audit data/daily.exits.parquet \
  --first-test 2024-01-01 --epochs 100

# Equal weights over the eligible assets, without training or optimization
uv run --no-sync python demo/ml/train.py --model equal_weight \
  --panel data/daily.npz --exit-audit data/daily.exits.parquet \
  --first-test 2024-01-01
```

Useful options:

- `--cost VALUE`: turnover cost coefficient (default `0`).
- `--smoke --epochs 1`: short check with reduced windows and date counts.
- `--help`: show all command-line options.

`--no-e2e` also supports the point baseline. To skip pretraining for E2E, set
`pretrain=False` in the demo's `run()` call. Model constants are set in Python.

## Ensembles

Add `--members 5` to a Gaussian command to train five independently initialized
networks per window. The default is one network; `--no-e2e` also works with ensembles.

```sh
uv run --no-sync python demo/ml/train.py --model gaussian --members 5 \
  --panel data/daily.npz --exit-audit data/daily.exits.parquet \
  --first-test 2024-01-01 --epochs 100
```

Members train separately on the same dates. Their equally weighted mixture supplies
one portfolio using average member covariance plus covariance from mean disagreement.
Forecast scores use the mixture density and marginal mixture intervals. Logs identify
each member and seed; NPZ exports include member forecasts. Training time grows roughly
with the member count. Disagreement is not a calibrated parameter posterior.

## Training logs

By default, each run creates a folder such as
`outputs/2026-11-03_14-30-00-123456_+0800_gaussian_e2e/`, using local time and the
model/training type. The summary CSV and log are saved together. Gaussian runs
also save an NPZ containing forecasts and portfolio weights in the same folder.

Each run prints the path to a timestamped JSONL log containing settings, split dates,
epoch losses, and completion status.

```sh
uv run --no-sync python demo/ml/plot_loss.py outputs/<run-folder>/<log>.jsonl
```

The plot is saved beside the log as `<run>.loss.png`. It can also be generated while
training is in progress.

## Interpretation

Changing-universe Gaussian members use diagonal covariance; ensembles also include
covariance from member disagreement. Fixed panels can use historical correlations. Predicted variances describe returns, not parameter
uncertainty. Portfolio dates require complete, aligned outcomes and confirmed exits.

Market/JKP targets are supplied-price returns; currency and total-return conventions
must be established for the input data. Turnover ignores return-induced weight drift,
and summaries use 252 intervals per year even when the calendar excludes dates.

## Tests

```sh
uv run --no-sync pytest -q
```
