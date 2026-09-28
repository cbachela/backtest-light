# btlight
The backtesting light (**btlight**) package is a lightweight library for equity portfolio optimization and backtesting produced for educational purposes.

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
uv run python demo/ml/wang_e2e.py --smoke --epochs 1 --risk variance --output outputs/comparison/wang_e2e.csv
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
selects the mean–variance QP; the default is standard-deviation risk. Both models
use a 10% per-asset cap and 252-day diagonal market-relative risk estimates.
The dense panel requires complete observations and uses single-day targets two
trading dates after the signal. It does not yet handle a changing stock universe.

BL remains optional and uses fixed uncertainty: `--bl constant` or
`--bl historical` (training residual variances frozen after pretraining).
Real-data BL additionally requires `--bl-market-weights PATH --bl-factors PATH`.
Market-weight CSVs must contain availability dates and columns matching the panel
assets exactly; the latest known row is used. The reverse-equilibrium prior uses
full excess-return covariance, while portfolio risk retains the baseline estimate.
`--uncertainty-output outputs/errors.npz` saves rolling residual diagnostics.
The learned heteroskedastic head discussed in the thesis is not implemented yet.

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
