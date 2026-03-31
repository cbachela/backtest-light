# flake8: noqa
# pylint: skip-file

################################
# Control Params
################################

label_recompute = False
feature_recompute = False
start_date = "2000-01-01"
end_date = "2023-01-01"

################################
# Logging
################################
import logging
from rich.console import Console
from rich.logging import RichHandler

# simple logging for this experiment script
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True)],
)
logger = logging.getLogger(__name__)


################################
# Data paths
################################
from pathlib import Path
import pandas as pd

# basic paths
repo_root = Path(__file__).resolve().parents[2]
data_path = repo_root / "data" / "Compustat" / "switzerland"


################################
# Data Alignment and 
# Daily Returns Computation
################################
import subprocess
subprocess.run(["python", "data_prep.py"], check=True)


################################
# Paths for generated data
################################

signal_path = data_path / "signals.parquet"
return_series_path = data_path / "return_series.parquet"

# generated input to training
feature_path = str(data_path / "features.parquet")
label_path = str(data_path / "labels.parquet")

# generated predictions
prediction_path = str(data_path / "ml_signal.parquet")
shap_path = str(data_path / "shap_values.parquet")



################################
# Features
################################
import pandas as pd
from btlight.ml.utils.format import check_if_multiindex, ensure_datetime_index

# only True for testing purposes to make the code fast
downsample = False

if Path(feature_path).exists() and not feature_recompute:

    logger.info(f"Loading features from {feature_path}")
    X = pd.read_parquet(feature_path)

else:

    logger.info(f"Preparing features from raw data in {data_path}")

    feature_cols = [
        "ret_6_1",      # Momentum6
        "ret_12_1",     # Momentum12  
        "qmj",          # Quality minus Junk
        "qmj_growth",   # GrowthComp
        "qmj_safety",   # SafetyComp
        "gp_at",        # GrossProfit
        "op_at",        # OpProfit
        "be_me",        # BookValue
        "debt_me",      # Leverage
        "at_gr1",       # AssetGrowth
        "oaccruals_at", # Accruals
    ]

    # load signals
    # X = pd.read_parquet(signal_path)
    X = pd.read_parquet(path=data_path / "jkp_data.parquet")
    X = X[feature_cols]

    X.index.names = ["DATE", "ID"]

    # keep only numeric columns (for now)
    X = X.select_dtypes(include="number")

    # Todo: one hot encode string features...

    # forward fill, 1y max
    X = X.groupby(level="ID").ffill(limit=4)

    # only for the example, in practice more careful
    X = X.dropna()

    # find first DATE with more than 50 unique IDs and no NAs
    dates_id_counts = X.dropna().groupby(level="DATE").size()
    start_date_id = dates_id_counts[dates_id_counts > 50].index[0]
    logger.warning(f"Start date (first DATE with >50 IDs): {start_date_id}")

    X = X.loc[X.index.get_level_values("DATE") >= start_date_id]

    # remove duplicates
    X = X[~X.index.duplicated(keep="last")]

    # check if we have the proper panel format
    check_if_multiindex(X)

    # check if date level is datetime, if not convert
    X = ensure_datetime_index(X)

    # sort
    X = X.sort_index()

    # assess the sample size over time
    # X.groupby(level="DATE").count().plot(title="Number of IDs per DATE")

    ################################
    # DownSampling
    ################################

    if downsample:
        # Find IDs present on EVERY date
        id_counts = X.index.get_level_values("ID").value_counts()

        time_grid = X.index.get_level_values("DATE").unique().sort_values()

        # First 10
        consistent_ids = id_counts[id_counts == len(time_grid)].index[:10]

        # Filter to only those IDs
        id_mask = X.index.get_level_values("ID").isin(consistent_ids)
        X = X[id_mask]

logger.info(f"Features prepared: X.shape={getattr(X, 'shape', None)}")


################################
# Label Creation
################################
import pandas as pd
import numpy as np
from btlight.ml.utils.grid import compound_returns_on_grid, shift_grid_returns
from btlight.ml.utils.format import check_if_multiindex, ensure_datetime_index

# time grid for label creation
time_grid = X.index.get_level_values("DATE").unique().sort_values()

# if label file exists, load it otherwise compute and persist
if Path(label_path).exists() and not label_recompute:

    logger.info(f"Loading labels from {label_path}")

    # load the label
    y_df = pd.read_parquet(label_path)

    # downstream needs a series
    y = y_df.squeeze()

else:

    logger.info(f"Preparing labels from raw data in {data_path}")
    # on s3 we still have multi columns

    return_series = pd.read_parquet(return_series_path, columns=["tot_return_gross"])
    # return_series = pd.read_parquet(return_series_path)

    # check if we have the proper panel format
    check_if_multiindex(return_series)

    # check if date level is datetime, if not convert
    return_series = ensure_datetime_index(return_series)

    # daily returns
    daily_ret = return_series.sort_index()

    # grid returns
    grid_ret = compound_returns_on_grid(
        return_series=daily_ret.squeeze(),
        time_grid=time_grid,
    )

    # shift back one unit on the grid
    y = shift_grid_returns(grid_ret, shift=-1)

    # remove duplicates
    y = y[~y.index.duplicated(keep="last")]

    ################################
    # Label Transformation
    ################################
    from btlight.ml.transformers.pipeline import TransformPipeline
    from btlight.ml.transformers.panel.cross_sectional import (
        CrossSectionalWinsorize,
        CrossSectionalZScore,
        CrossSectionalClip,
        CrossSectionalPIT,
    )

    # label transformation pipeline
    label_pipeline = TransformPipeline(
        [
            # CrossSectionalZScore(),
            CrossSectionalWinsorize(lower=0.01, upper=0.99),
            CrossSectionalPIT(),
        ]
    )

    y = label_pipeline.fit_transform(y)
    y = y.dropna()

logger.info(f"Labels prepared: y.shape={getattr(y, 'shape', None)}")

# check the return dist over time
# _y_df = y.unstack()
# _y_df.max(axis=1).plot()
# _y_df.mean(axis=1).plot()
# _y_df.min(axis=1).plot()
# import matplotlib.pyplot as plt
# plt.hist(_y_df.tail(1).values[0], bins=20)
# _y_df.quantile(0.99, axis=1).plot()
# _y_df.mean(axis=1).plot()
# _y_df.quantile(0.01, axis=1).plot()


################################
# Alignment of X and y
################################

common_index = X.index.intersection(y.index).sort_values()
X = X.loc[common_index]
y = y.loc[common_index]

print("X.shape", X.shape)
print("y.shape", y.shape)

assert X.index.equals(y.index)


# persist the aligned
if not Path(feature_path).exists() or feature_recompute:
    print("Persisting Features")
    X.to_parquet(feature_path)

# persist the label for future use
if not Path(label_path).exists() or label_recompute:
    print("Persisting Labels")
    y.to_frame("return").to_parquet(label_path)


logger.info(f"Aligned X and y: X.shape={X.shape}, y.shape={y.shape}")


################################
# Define a model / pipeline
################################
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from btlight.ml.metrics.scoring import ic_score_func


ic_score = make_scorer(ic_score_func, greater_is_better=True)

# linear model
# pipeline = Pipeline(
#     [
#         # not really needed since our input is very tamed
#         #("scaler", MinMaxScaler()),
#         # add squared features
#         #("poly", PolynomialFeatures(degree=2, include_bias=False)),
#         # simple regressor
#         #("pca", PCA()),
#         ("regressor", Ridge(random_state=42))
#     ]
# )

# param_grid = {"regressor__alpha": [1e-6, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e6]}
# explainer_type = "linear"


# # Neural Net (better to use pytorch if you want transformers etc.)
# # skorch is a good wrapper to keep the below framework
# from sklearn.neural_network import MLPRegressor

# pipeline = Pipeline(
#     [
#         # MLP is sensitive to feature scale
#         ("scaler", StandardScaler()),
#         (
#             "regressor",
#             MLPRegressor(
#                 hidden_layer_sizes=(64, 32),
#                 activation="relu",
#                 solver="adam",
#                 max_iter=200,
#                 early_stopping=True,
#                 validation_fraction=0.1,
#                 random_state=42,
#             ),
#         ),
#     ]
# )

# param_grid = {
#     "regressor__hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16)],
#     "regressor__alpha": [1e-4, 1e-3, 1e-2],  # L2 regularization
#     "regressor__learning_rate_init": [1e-3, 5e-4],
# }

# XGBoost model
from xgboost import XGBRegressor
pipeline = Pipeline(
    [
        (
            "regressor",
            XGBRegressor(
                objective="reg:absoluteerror",
                random_state=42,
                n_estimators=200,
                tree_method="hist",
            ),
        ),
    ]
)
param_grid = {
    "regressor__max_depth": [2, 3],
    "regressor__learning_rate": [0.01, 0.02],
    "regressor__n_estimators": [10, 50],
    "regressor__reg_alpha": [0.5, 1.0, 2.0],      # L1 regularization
    "regressor__reg_lambda": [1.0, 5.0, 10.0],    # L2 regularization
    "regressor__min_child_weight": [5, 10],       # Prevent small leaves
    "regressor__subsample": [0.6, 0.8],           # Row subsampling
    "regressor__colsample_bytree": [0.6, 0.8],   # Column subsampling
}
explainer_type = "tree"


# # more complex pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler
# from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
# from sklearn.decomposition import PCA

# pipeline = Pipeline(
#     [
#         # guard against unseen nans at inference; median robust to outliers
#         ("imputer", SimpleImputer(strategy="median")),
#         # drop near-constant features before scaling
#         ("variance_filter", VarianceThreshold(threshold=1e-4)),
#         # iqr-based scaling; fat-tailed signals (momentum etc.) break std-based scalers
#         ("scaler", RobustScaler(quantile_range=(5.0, 95.0))),
#         # univariate linear filter; k swept in grid so the filter can be disabled
#         ("feature_selection", SelectKBest(score_func=f_regression, k=8)),
#         # decorrelates redundant features, n_components=None = passthrough
#         ("pca", PCA(n_components=None, whiten=True)),
#         (
#             "regressor",
#             XGBRegressor(
#                 objective="reg:absoluteerror",
#                 tree_method="hist",
#                 random_state=42,
#                 n_estimators=500,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 reg_alpha=0.1,       # l1: sparsifies feature usage across trees
#                 reg_lambda=5.0,      # l2: shrinks leaf weights, both can coexist
#                 min_child_weight=5,
#                 gamma=0.1,
#             ),
#         ),
#     ]
# )

# param_grid = {
#     # data-prep
#     "imputer__strategy": ["median", "mean"],
#     "feature_selection__k": [6, 8, "all"], # "all" disables the filter
#     "pca__n_components": [None, 5],          # None = pca off

#     # tree structure
#     "regressor__max_depth": [3, 4, 5],
#     "regressor__learning_rate": [0.01, 0.03, 0.05],

#     # stochastic subsampling
#     "regressor__subsample": [0.7, 0.85, 1.0],
#     "regressor__colsample_bytree": [0.7, 0.85, 1.0],

#     # regularisation — l1 and l2 are independent and swept jointly
#     "regressor__reg_alpha": [0.0, 0.1, 1.0],
#     "regressor__reg_lambda": [1.0, 5.0, 10.0],
#     "regressor__min_child_weight": [3, 5, 10],
#     "regressor__gamma": [0.0, 0.1, 0.5],
# }

# Todo: implement XGBRanker if you want to do direct ranking optimization, 
# from xgboost import XGBRanker
# pipeline = Pipeline(
#     [("regressor", XGBRanker(objective='rank:ndcg'))]
# )



####################################################
# Grid Search
####################################################
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold
from btlight.ml.splitters.rolling_timeseries_split import PanelTimeSeriesSplit


grid_search = GridSearchCV(
    pipeline,
    param_grid,
    # cv=KFold(n_splits=5),
    cv=PanelTimeSeriesSplit(n_splits=4, date_level="DATE"),
    n_jobs=-1,
    scoring=ic_score,
    # scoring="neg_mean_squared_error",
    refit=True,
)

####################################################
# Time Grid for Rolling Split
####################################################

# this can only be done if X and y are aligned
time_grid = X.index.get_level_values("DATE").unique().sort_values()

# to reduce the sample size for testing purposes,
# we can filter the time grid to start from a later date
# also the compustat data seem poor in the very last years
if start_date is not None:
    time_grid = time_grid[time_grid > start_date]
if end_date is not None:
    time_grid = time_grid[time_grid < end_date]


####################################################
# Train and Test Rolling Split
####################################################
from btlight.ml.splitters.rolling_timeseries_split import ObservationGridRollingSplit

rolling_splitter = ObservationGridRollingSplit(
    observation_dates=time_grid,
    train_window_obs=12,
    skip_obs_between_train_test=0,
    retrain_stride=1,
)

rolling_splitter.print_splits(X=X)


####################################################
# Understanding Training / Validation Splits
####################################################

# todo: add sector validation, add cpcv and embargo.
#splits = rolling_splitter.split(X=X)

# get training info of first split
# first split
#train_idx, test_idx = next(splits)

#PanelTimeSeriesSplit(n_splits=5, date_level="DATE").show(X=X.loc[train_idx])


####################################################
# Train (will persist the models to disk)
####################################################
from btlight.ml.training.traintest import train_func
from sklearn.base import clone
import joblib


n_jobs = joblib.effective_n_jobs()

# if already trained will not do it again
# run the training of the models in parallel, this will take time
with joblib.parallel_backend("loky", n_jobs=n_jobs, verbose=True):
    jobs = []
    jobs.extend(
        joblib.delayed(train_func)(
            # train_func(
            model=clone(grid_search),
            X=X,
            y=y,
            train_idx=train_idx,
            # if you want a model by asset, pass y as a wide dataframe and
            # loop over assets here with target_asset=asset_name
            # target_asset=target_asset,
        )
        for train_idx, _ in rolling_splitter.split(X=X)
    )

    joblib.Parallel()(jobs)


logger.info("Training Done")


####################################################
# Test (will load the models from disk and generate predictions)
####################################################
import joblib
from btlight.ml.training.traintest import test_func

n_jobs = joblib.effective_n_jobs()
# n_jobs = 50

# run the test, which is simply generating the predictions
# here also in parallel
with joblib.parallel_backend("loky", n_jobs=n_jobs, verbose=True):
    jobs = []
    jobs.extend(
        joblib.delayed(test_func)(
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        for train_idx, test_idx in rolling_splitter.split(X=X)
    )

    # list of tuples (resolved_target, y_pred_test, y_pred_train)
    result = joblib.Parallel()(jobs)

logger.info("Testing Done")

####################################################
# Persist the predictions
####################################################

# build dataframe from y_pred_test (second element of tuple)
# then unstack to have a dataframe with index DATE and columns ID, values are predictions
y_pred_test_df = pd.concat(
    [item[1] for item in result],
    axis=0,
).unstack()

print(y_pred_test_df.head())
print(y_pred_test_df.index)

row_non_na_counts = y_pred_test_df.count(axis=1)
print("# of predictions per unique DATE:", row_non_na_counts)

y_pred_test_df.to_parquet(prediction_path)

print(f"Stored Predictions in {prediction_path}")
print("Predictions Done")
logger.info(f"Predictions written to {prediction_path}")


####################################################
# Shap Values Computation
####################################################
import joblib
import pandas as pd
from btlight.ml.explain.xai import get_shaply_values

n_jobs = joblib.effective_n_jobs()
# n_jobs = 50

# run the test, which is simply generating the predictions
# here also in parallel
with joblib.parallel_backend("loky", n_jobs=n_jobs, verbose=True):
    jobs = []
    jobs.extend(
        joblib.delayed(get_shaply_values)(
            X=X,
            y=y,
            train_idx=train_idx,
            target_asset=None,
            explainer_type=explainer_type,
        )
        for train_idx, _ in rolling_splitter.split(X=X)
    )

    # list of tuples (resolved_target, y_pred_test, y_pred_train)
    result = joblib.Parallel()(jobs)

# collect results and persist in redable form
dfs = []

for i, sv in enumerate(result):
    df = pd.DataFrame(
        sv.values,
        columns=sv.feature_names,
        # index=X.index # poor solution, pipeline might have dropped rows
    )
    df["split"] = i
    dfs.append(df)

df_all_shap = pd.concat(dfs)
df_all_shap.to_parquet(shap_path)


logger.info(f"Saved SHAP values to {shap_path}")


import pdb
pdb.set_trace()
# end of computational part - below is analysis and understandings.

####################################################
# Shap Value Analysis
####################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# java script based visualizations, only works in jupyter.
shap.initjs()


# load all shap values
df_all_shap = pd.read_parquet(shap_path)

importance = df_all_shap.drop(columns="split").abs().groupby(df_all_shap["split"]).mean()

# feature importance plot through time / the splits currently
importance.cumsum().plot(figsize=(12, 6))
plt.grid(alpha=0.4)
plt.show()

importance.mean().sort_values(ascending=False)


####################################################
# Metric Analysis on Test Predictions
####################################################
import pandas as pd
import matplotlib.pyplot as plt
from btlight.ml.metrics.scoring import ic_score_func, spearman_correlation_per_date, mae_per_date

y_hat = pd.read_parquet(prediction_path).stack().dropna()
y = pd.read_parquet(label_path).squeeze()
X = pd.read_parquet(feature_path)


# Compute metrics and store in a dataframe
metrics_df = pd.DataFrame({
    "trained_model": spearman_correlation_per_date(y_true=y, y_pred=y_hat),
    "last_observation": spearman_correlation_per_date(y_true=y, y_pred=y.groupby(level="ID").shift(1).dropna()),
    "qmj": spearman_correlation_per_date(y_true=y, y_pred=X['qmj']),
})

# Plot cumulative sum
metrics_df.dropna().cumsum().plot(figsize=(6, 4))
plt.title("Test Sample Performance")
plt.ylabel(r"rank spearman $f(y,\hat{y})$")
plt.legend(["trained model", "last observation as prediction", "qmj"], loc="upper left")
plt.grid(alpha=0.4)
plt.show()

# Compute MAE metrics
y_sf = X['qmj']

# Scale qmj feature by date
y_sf_date_scaled = y_sf.groupby(level="DATE").transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

metrics_df2 = pd.DataFrame({
    "trained_model": mae_per_date(y_true=y, y_pred=y_hat),
    "last_observation": mae_per_date(y_true=y, y_pred=y.groupby(level="ID").shift(1).dropna()),
    "qmj": mae_per_date(y_true=y, y_pred=y_sf_date_scaled),
})

# Plot MAE
metrics_df2.dropna().plot(figsize=(6, 4))
plt.title("Test Sample Performance - MAE")
plt.ylabel("Mean Absolute Error")
plt.legend(["trained model", "last observation as prediction", "qmj"], loc="upper left")
plt.grid(alpha=0.4)
plt.show()


####################################################
# Train and Test Error
####################################################
import matplotlib.pyplot as plt

# train and test evaluation
ic_train = []
ic_test = []
for item in result:
    _, y_hat_test, y_hat_train = item

    ic_test.append(ic_score_func(y.reindex(y_hat_test.index), y_hat_test.values.reshape(-1)))
    ic_train.append(ic_score_func(y.reindex(y_hat_train.index), y_hat_train.values.reshape(-1)))


plt.plot((np.array(ic_test)), label="IC Test")
plt.plot((np.array(ic_train)), label="IC Train")
plt.legend()
plt.grid(alpha=0.4)
plt.show()

y.reindex(y_hat.index), y_hat
ic_score_func(y.reindex(y_hat.index), y_hat.values.reshape(-1))
pd.Series(y_hat.values.reshape(-1))



####################################################
# look at one example / split
####################################################
study = False

if study:

    splits = rolling_splitter.split(X=X)

    # get training info of first split
    # first split
    train_idx, test_idx = next(splits)
    # this is how you move to the next split
    train_idx, test_idx = next(splits)

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    # clone model and fit
    model = clone(grid_search)
    model.fit(X=X_train, y=y_train)

    # results from GridSearchCV
    results = model.cv_results_

    # Find all hyperparameter columns
    param_cols = [c for c in results.keys() if c.startswith("param_regressor__")]

    # extract values for each parameter
    param_data = {}
    for c in param_cols:
        values = np.array(results[c])
        try:
            values = values.astype(float)
        except:
            pass
        param_data[c.replace("param_regressor__", "")] = values

    # add the test score (negative mse, higher is better)
    param_data["mean_test_score"] = results["mean_test_score"]
    param_data["std_test_score"] = results["std_test_score"]
    # ad the mse (sign flip, note this will need chagnes if you change the score func.)
    param_data["mean_test_mse"] = -results["mean_test_score"]

    # Make DataFrame
    df = pd.DataFrame(param_data)

    # sorting
    param_name = "alpha"  # or alpha or what is suitable
    df = df.sort_values(by=[param_name], ascending=False)
    # df = df.sort_values(by=["alpha"], ascending=False)

    # simple Hyperparam plot for Ridge Regression
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.semilogx(df[param_name], df["mean_test_score"], marker="o", linestyle="-")
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Mean CV Score")
    plt.title("Ridge: CV Score vs Alpha")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    # why is the standard deviation so high?
    plt.figure(figsize=(8, 5))

    # Semilog-x plot with shaded std region
    plt.semilogx(
        df[param_name], df["mean_test_score"], marker="o", linestyle="-", label="Mean CV MSE"
    )
    plt.fill_between(
        df[param_name],
        df["mean_test_score"] - df["std_test_score"],
        df["mean_test_score"] + df["std_test_score"],
        alpha=0.2,
    )

    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Mean CV Score")
    plt.title("Ridge: CV Score vs Alpha")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    print(
        pd.Series(
            model.best_estimator_.named_steps["regressor"].coef_, index=X.columns
        ).sort_values()
    )
