# btlight
The backtesting light (**btlight**) package is a lightweight library for equity portfolio optimization and backtesting produced for educational purposes.

## Daily market and JKP inputs

Run from the repository root. `prepare_daily.py` joins the supervisor's market
and JKP Parquet files and writes a daily panel. The defaults use signals from
2020 through 2024, loading earlier observations for warm-up and later observations
for labels. Change `--start` and `--end` to select another period.

```sh
uv run --no-sync python demo/data/prepare_daily.py --split-corrections data/split_corrections.parquet --terminal-events data/corporate_actions.parquet --output data/daily.npz
```

The prepared file already exists locally. Use `--overwrite` to rebuild it and its
audits, or choose another output filename. Alongside the panel it saves
`daily.json` (settings/counts), `daily.audit.parquet` (monthly matches and targets),
`daily.returns.parquet` (extreme observed returns) and `daily.exits.parquet`
(candidate eligibility exits with next-close quotes), and `daily.calendar.parquet`
(country/date inactivity evidence).

Monthly records join by company and country, backward from month-end plus one
calendar day, with maximum age 62 days. This is an availability assumption, not
verified source vintage metadata. Multiple issues in the same company/country/month
produce missing characteristics, not an arbitrary selected listing. Company IDs
identify the supplied daily price series; they do not establish its share-class
history. The ambiguous index/data column named `id` in JKP is handled separately.

There are 25 inputs: seven daily return/volatility signals, supplied liquidity,
eight JKP characteristics (`be_me`, `op_at`, `gp_at`, `ocf_at`, `cash_at`, `debt_at`,
`at_gr1`, `sale_gr1`), and nine missing flags for liquidity/JKP. Signals use date-wise
inverse-normal ranks, with neutral zero for missing optional scores. Daily market
history requires 63 consecutive usable returns. Feature names live in
`src/btlight/data/daily.py`. Future prices never decide membership or feature values.

Targets are **supplied-price returns**, not asserted EUR or total returns. The
source currency and adjustment conventions are unknown. The supplied universe is
retained subject to market-history requirements; the older dataset's additional
20% size screen is not reapplied across unverified capitalization currencies.
The schedule excludes dates with country-wide inactivity: at least five previously
quoted companies and >=95% missing quotes or zero supplied liquidity in any country.
No portfolio rebalance occurs on these inferred closure dates; holding-period
returns span the skipped days. This is not an official exchange calendar. Entry
and exit use successive remaining closes. Unknown prices are not carried forward.
All finite returns are retained in targets and risk history. Absolute returns above
50% are review flags only; they do not remove observations or cap returns. Rank
normalization applies to inputs, not realized portfolio returns. Split matching
converts both Compustat endpoints to EUR before comparing supplied prices. The
current build applies 96 split/distribution adjustments, including Vivendi spin-offs
and DMGT’s special distribution. Distributed shares use a same-close cash-equivalent
approximation; DMGT’s delayed share delivery is treated as immediate.

The reviewed `corporate_actions.parquet` encodes 69 corporate transactions and
11 explicit zero-recovery assumptions. Restoring histories added Iliad, DMGT and
Credit Suisse outcomes, plus assumed zero recovery for Sanochemia, NMC, Intu,
Steinhoff, Philipp Holzmann and SAS. These six assumptions use source-series endings
as timing proxies, not independently verified zero payouts. Cash consideration is converted to
EUR; share consideration uses the exchange ratio and successor price. This baseline
converts proceeds to cash at the scheduled rebalance, ignoring settlement delays,
taxes and fractional-share costs. Some stop dates use the supplied series ending;
the file records each timing/election assumption and source. Avast's successor leg
uses a rounded acquisition-date valuation from its SEC filing. Terminal anchors
are checked against source prices and local Compustat currency conversion.
Stopped securities cannot re-enter. Holdings crossing a zero-recovery date receive
-100%; new orders reaching a stopped security are cancelled and stay in cash,
identified by `target_basis=cancelled_order_cash` in the audit. Terminal distributions
are accepted by the exit loader, including zero-valued write-offs. These are modelled
settlements, not claims that a market sale took place.

## Run the models

Every `train.py` run creates a timestamped JSON Lines log beside the summary CSV
and prints its path. Use `--log outputs/gaussian_run1.jsonl` to choose a path;
existing logs are never overwritten. Records are flushed immediately. They include
the model/data/output paths, actual model settings (including initial variance,
variance floor and NLL weight), architecture, split dates, optimizer settings,
UTC timestamps, elapsed time, epoch duration, training/validation losses, best epoch,
early stopping, and completion or failure. `epoch_start` marks work in progress;
`epoch` contains the completed epoch's losses. Already-running processes do not
pick up this logging change.

```sh
tail -f outputs/gaussian_run1.jsonl
uv run --no-sync python demo/ml/plot_loss.py outputs/gaussian_run1.jsonl
```

The plot is saved beside the log as `gaussian_run1.loss.png`; `--output PATH`
overrides that location. The script also works during training and ignores an
unfinished final log line. Each quarter/stage has its own panel: NLL or MSE
pretraining and the E2E objective have different scales. Training losses use
training-mode dropout/batch normalization; validation losses use evaluation mode.

`demo/ml/train.py` replaces the old `wang_e2e.py` entry point. Choose `--model gaussian`
for distributional forecasting (the default), or `--model baseline` for the
Wang-inspired point model. Both use `src/btlight/ml/training/e2e.py`.
Set `pretrain=False` in the demo's `run(...)` call to train directly from random
initialization; the default is `True`. Gaussian E2E retains its NLL term either way.

```sh
uv run --no-sync python demo/ml/train.py --model gaussian --panel data/daily.npz --exit-audit data/daily.exits.parquet --first-test 2024-01-01 --epochs 100 --output outputs/gaussian_e2e.csv --distribution-output outputs/gaussian_e2e.npz
uv run --no-sync python demo/ml/train.py --model baseline --panel data/daily.npz --exit-audit data/daily.exits.parquet --first-test 2024-01-01 --risk variance --epochs 100 --output outputs/baseline_e2e.csv
```

The rebuilt panel contains **940,497 rows, 1,198 dates and 845 companies**, with
**zero missing targets and zero unresolved exits**. All dates pass portfolio
validation and all membership exits have lookup values. There are 258 extreme-return
review rows, including warm-up observations; these are retained, not missing targets.
Both baseline and Gaussian E2E completed real-data checks on this same file using
`--risk variance --smoke --epochs 1`, producing `outputs/baseline_cleaning_smoke.csv`
and `outputs/gaussian_cleaning_smoke.csv`. These are short wiring checks, not full
training experiments or meaningful performance estimates. No general outlier
clipping or blanket zero-return filling is applied. Synthetic checks can also run
with `--smoke --epochs 1` and no `--panel` or `--exit-audit`.
Performance summaries still annualize at 252 intervals; this needs revisiting for
the reduced calendar before interpreting annualized results.

## Quick start (uv)

### 1) Install uv
- **Linux/macOS**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
- **Windows (PowerShell)**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex
    ```

### 2) Create a venv
```bash
uv venv
```

### 3) Sync dependencies
```bash
uv sync
```

### 5) Add a package
Below an example of how to add a package automatically to the package (directly into the `pyprocet.toml`)
```bash
uv add xgboost
```



## Thesis portfolio experiments

Reusable model/training code is in `src/btlight/ml`; data preparation is in
`src/btlight/data`. Differentiable portfolio and BL layers live in `optimization`
and `estimation`, alongside the existing NumPy implementations. They preserve
PyTorch gradients; the ordinary backtesting solvers do not. The Wang-specific
splitter retains the 39/9/3-month windows and purges unavailable labels.

The E2E extra requires Python 3.10 or newer; use Python 3.11 below. On Python
3.9 its dependencies are skipped, allowing the WRDS extra to remain usable.
Run from this repository's root:

```sh
uv sync --python 3.11 --extra e2e --extra wrds
uv run python demo/optimization/mean_variance.py --smoke --output outputs/comparison/mean_variance.csv
uv run python demo/ml/train.py --model baseline --smoke --epochs 1 --risk variance --output outputs/comparison/wang_e2e.csv
uv run python demo/ml/visualize.py outputs/comparison
uv run pytest -q
```

For the copied French panel, replace `--smoke` with
`--panel data/panel.npz --first-test 2015-01-01`. This starts a full experiment.
Rebuild the panel or download a stock sample with:

```sh
uv run python demo/data/prepare_french.py --factors data/F-F_Research_Data_Factors_daily.csv
uv run python demo/data/download_sample.py --config ../data-wrds/config.ini
```

The downloader defaults to AAPL/MSFT/KO, 2020–2024, and `data/wrds/`.
It retains the source downloader's USD conversion and fundamentals-issue filter;
the three-stock sample is a download check, not a model-ready panel or universe.
Credentials remain in the original local INI file. No connection occurs on import.

Wang runs MSE pretraining followed by decision-focused training. `--risk variance`
selects the mean–variance QP; the default is standard-deviation risk. Both models use
a 10% per-asset cap and 252-day diagonal market-relative risk estimates. The dense
panel requires complete observations and uses single-day targets two trading dates
after the signal. The changing-universe dataset uses packed date cross-sections (see
below).

The active Wang runner omits the sidelined BL update and residual diagnostics.
Standalone BL estimation components remain available for later research. Experiment
controls stay in the CLI; model constants live in the Python functions.

### Probabilistic returns (single network)

`--model gaussian` uses the existing shared backbone with separate mean and
positive return-variance heads, following Lakshminarayanan et al. (2017), section
2.2.1: https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-
uncertainty-estimation-using-deep-ensembles.pdf There is no ensemble, adversarial
training, parameter posterior or MAP prior. The scaled softplus variance starts at
1e-4, with a variance floor of 1e-8; these are daily decimal-return squared units.
Gaussian mean-head weights start small (standard deviation 1e-3) with zero bias.

The joint covariance is D R D: D holds predicted standard deviations; R is a
252-observation historical target-basis return correlation matrix, shrunk toward
identity by 0.1. This correlation model is our portfolio extension, not part of the
paper's scalar regression specification or a fitted DCC model. Targets retain the
panel's benchmark and horizon conventions.

Training uses Gaussian joint NLL per asset for pretraining, followed by portfolio
loss plus 0.001 times NLL (an uncalibrated pilot setting to tune on validation).
Allocation uses predicted covariance; the outer risk penalty uses the existing
historical diagonal covariance. This fixed risk proxy prevents directly reducing the
evaluation penalty by reporting lower risk, but does not guarantee calibrated
distributions. NLL pretraining is maximum likelihood; a standalone likelihood-only
comparison and MAP estimation remain future work.

Gaussian mode uses variance risk, selected automatically by the CLI. `--distribution-
output` saves means, marginal variances, covariance factors A (covariance = A.T @ A),
targets, weights and per-asset joint NLL. Summary CSV adds held-out NLL and pooled
marginal 95% interval coverage; coverage is not a joint or conditional calibration
test.

```sh
uv run --no-sync python demo/ml/train.py --smoke --epochs 1 --model gaussian --distribution-output outputs/gaussian_smoke.npz --output outputs/gaussian_smoke.csv
```

This is a wiring check, not a trained performance benchmark. No new dependencies
are needed beyond the existing E2E environment.

Historical result CSVs are copied to `outputs/legacy/`. New results go to `outputs/`;
keep comparable settings together when plotting. Local datasets, outputs, and
credentials are excluded from Git. These experiments are package-integrated but
still use their original walk-forward loop, not the `Backtest` orchestration API.


For a broad Compustat Global download (2000–2025):

```sh
uv run --extra wrds python demo/data/download_europe.py --config ../data-wrds/config.ini --inspect
uv run --extra wrds python demo/data/download_europe.py --config ../data-wrds/config.ini
```

Annual security and FX files go to `data/compustat_europe/`, alongside reference
snapshots and download settings. Rerunning resumes completed files; `--refresh`
replaces them. Different date ranges or filters require a new `--output-dir`.
The download keeps all issue types, inactive issues, and incomplete observations.
European candidates use incorporation/headquarters and available exchange-country
metadata (EU + UK, Switzerland, Norway, Iceland). Current headers can miss past
geographic affiliations: `--all-global --output-dir data/compustat_global` removes
geographic filtering entirely, at substantially greater download size.
If exchange-country metadata is absent, supply verified `--exchange-codes` or use
`--all-global`. Missing optional fields/tables are reported; use `--reference-tables`
for additional history tables discovered by `--inspect`.
Reference and FX columns, identifiers, dates, and category codes are stored as text;
daily prices, adjustments, shares and volume are numeric. No returns, currency
conversion, deduplication, or investment-universe filtering is performed yet.

### European changing-universe inputs (2026-10-18)

The cleaned size-screened data now has a practical forecast-training input:
`data/europe_dynamic_v1.npz`. `load_data` accepts this packed NPZ or the cleaned
directory. Existing fixed-universe NPZ files remain supported.

The starter specification has 16 signals plus eight accounting missingness flags:

- Market: `return_1`, `return_5`, `return_21`, `momentum_63_21`,
  `volatility_21`, `volatility_63`, `max_return_21`, `turnover_63`.
- Accounting: `income_assets`, `operating_income_assets`, `cashflow_assets`,
  `cash_assets`, `debt_assets`, `sales_assets`, `asset_growth`, `liabilities_assets`.

Use existing `_normal` scores from the size-screened cross-section. Keep existing
coverage-aware sector/date imputations. Remaining unavailable accounting scores
receive a neutral zero with a missingness flag; this is an explicit model encoding,
including structural/stale missingness, not an assertion that an accounting ratio
equals zero. Raw source values remain unchanged. Market signals are never filled:
all eight must be finite, requiring roughly 63 uninterrupted usable sessions.
Dates need at least ten eligible companies. The 126/252-session signals and several
less broadly applicable accounting ratios are deferred; this is a starter choice,
not an empirically optimized final feature set.

`DynamicPanel.day(t)` selects packed rows for that date; `asset_index` maps them to
stable `gvkey:iid` identifiers. A company can enter, leave and re-enter without
being required to survive through the sample. Source histories are retained before
selection. `risk_variance` records the available 63-session daily variance for later
portfolio integration. Missing targets stay NaN and never determine eligibility.
`label_mask(t, before=...)` applies per-security availability and split purging.
Labels remain EUR total returns from entry close t+1 to exit close t+2 on the
security's exchange sessions, so label dates can differ across countries.

```sh
uv run --no-sync python -m btlight.data.panel --source ../data-wrds/Data/europe_daily_2020_2024_size80 --output data/europe_dynamic_v1.npz
uv run --no-sync python demo/ml/train.py --panel data/europe_dynamic_v1.npz --model gaussian --first-test 2024-01-01 --smoke --epochs 1 --output outputs/europe_dynamic_smoke.csv --distribution-output outputs/europe_dynamic_smoke.npz
```

Preparation refuses to overwrite an existing output. The runner uses the same shared
network and E2E loop for fixed and changing universes. Date batches can have different
security IDs and counts. No sample-wide survivor intersection is needed.

Holdings are aligned by security ID. New securities start with zero holdings; a size
exclusion forces a sale only with a confirmed executable quote. Both the departing
weight and new purchases contribute to turnover. The simplified Wang convention
ignores return-induced weight drift. Supply the size-exit audit for changing-universe
runs with departures:

```sh
uv run --no-sync python demo/ml/train.py --panel data/europe_dynamic_v1.npz --exit-audit ../data-wrds/Data/europe_daily_2020_2024_size80/size_exits.parquet --model gaussian --first-test 2024-01-01 --smoke --epochs 1
```

Dynamic risk uses the loader's known 63-session volatility estimates, with diagonal
covariance. Gaussian allocation learns the marginal variances and assumes zero
cross-asset correlations. This differs from the fixed-panel Gaussian correlation
proxy described above. Vector risk inputs avoid dense covariance allocation.

**The current European NPZ still contains missing and asynchronous outcomes.** The
runner rejects such inputs explicitly; the command above cannot yet produce a valid
full-Europe portfolio backtest. Every selected date needs complete outcomes, a common
execution interval, and continuity into the next interval. Missing holdings need an
explicit exit or retention treatment. Delayed exit quotes remain unresolved by the
simple immediate-exit lookup. No future-label membership filtering or zero-return
imputation is performed. Synthetic changing-universe E2E checks cover both prediction
models, entry/exit/re-entry and costs; they are implementation checks, not empirical
performance evidence.

The separate forecast-only runner was removed to keep one training path focused on
E2E. Existing NPZ inputs and their missing-label metadata remain unchanged. This
older European panel does not use JKP; the daily supervisor-data adapter above does.
