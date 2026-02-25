(btlight.ml — Machine learning helpers)

This package contains small, focused utilities used to prepare data, build labels,
split time series for backtesting, and run lightweight training/evaluation flows.

Subpackages
- `explain`: Tools and helpers for model explanation and interpretability (feature
	importance, partial dependence, SHAP-like wrappers). Use these utilities to
	generate human-readable explanations of model outputs.
- `features`: Functions to build and transform raw input features from market and
	fundamental data. Contains feature engineering helpers such as aggregations,
	z-score normalization, and cross-sectional transforms.
- `io`: Input/output helpers for reading/writing ML artifacts and datasets
	(small CSV/parquet loaders, lightweight serialization helpers). Keeps file
	handling consistent across examples and tests.
- `labels`: Utilities to create target variables for supervised tasks (returns,
	ranking labels, classification thresholds). Provides canonical label builders
	used throughout the examples and experiments.
- `metrics`: Lightweight evaluation metrics and wrappers for easy scoring of
	predictions (e.g., NDCG, grouped ranking metrics, and convenience wrappers
	that handle missing values and grouping by date).
- `naming`: Centralized naming conventions and simple helpers for consistent
	column/index names across feature/label pipelines.
- `splitters`: Time-series-aware splitters (rolling/expanding window cross-
	validation) designed for backtesting scenarios where data cannot be shuffled.
	Includes functions to generate train/validation/test index sets.
- `training`: Small training utilities and helper routines for running model
	fits and simple experiment loops (train/test harnesses, lightweight logging,
	and reproducibility helpers).
- `transformers`: Reusable data transformers (scalers, encoders, grouped
	operations) that follow a simple fit/transform pattern for pipeline usage.
- `utils`: Miscellaneous helpers (date utilities, small vectorized operations,
	and sanity-check functions) that support the other subpackages.

If you want, I can also add a short example showing a minimal pipeline that
uses `splitters`, `features`, `labels`, and `training` together. 
