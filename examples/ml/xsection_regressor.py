################################
# Control Params
################################

label_recompute = False
feature_recompute = False

################################
# Logging
################################
import logging

# simple logging for this experiment script
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


################################
# Data paths
################################
from pathlib import Path
import pandas as pd

# raw input
repo_root = Path(__file__).resolve().parents[2]
data_path = repo_root / "data" / "DMEXLME" / "20251231"
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
downsample = True

if Path(feature_path).exists() and not feature_recompute:

    X = pd.read_parquet(feature_path)

else:

    # load signals
    X = pd.read_parquet(signal_path)

    # inconsistent naming in signal service with lowercase forces us to
    # todo: this shouldb't happen blindly
    X.index.names = ["DATE", "ID"]

    # check if we have the proper panel format
    check_if_multiindex(X)

    # check if date level is datetime, if not convert
    X = ensure_datetime_index(X)

    # sort
    X = X.sort_index()

    # keep only numeric columns (for now)
    X = X.select_dtypes(include="number")

    # in a first shot allow this drastic feature reduction
    # selection = ["profitability", "growth_qa", "momentum", "value_sector_stdz", "volatility", "safety", "investment"]
    # selection = ["profitability", "value"]
    # X = X[selection]

    # # we can add more features, below likely redundant:
    # for col in X.columns:
    #     X[f"{col}_rank"] = X.groupby("DATE")[col].rank(pct=True)
    # can also operate on string / hot encoded features if available.

    # this is dangerous but okay for the example
    X = X.dropna()

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

    # load the label
    y_df = pd.read_parquet(label_path)

    # downstream needs a series
    y = y_df.squeeze()

else:
    # on s3 we still have multi columns
    return_series = pd.read_parquet(return_series_path, columns=["tot_return_gross"])
    # return_series = pd.read_parquet(return_series_path)

    return_series = return_series.astype(float).squeeze()

    # check if we have the proper panel format
    check_if_multiindex(return_series)

    # check if date level is datetime, if not convert
    return_series = ensure_datetime_index(return_series)

    # daily returns
    daily_ret = return_series.sort_index()

    # grid returns
    grid_ret = compound_returns_on_grid(
        return_series=daily_ret,
        time_grid=time_grid,
    )

    # shift back one unit on the grid
    y = shift_grid_returns(grid_ret, shift=-1)

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

common_index = X.index.intersection(y.index)
X = X.loc[common_index].sort_index()
y = y.loc[common_index].sort_index()

print("X.shape", X.shape)
print("y.shape", y.shape)


# persist the aligned
if not Path(feature_path).exists() or feature_recompute:
    print("Persisting Features")
    X.to_parquet(feature_path)

# persist the label for future use
if not Path(label_path).exists() or label_recompute:
    print("Persisting Labels")
    y.to_frame("return").to_parquet(label_path)


assert X.index.equals(y.index)

logger.info(f"Aligned X and y: X.shape={X.shape}, y.shape={y.shape}")


################################
# Define a model / pipeline
################################
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from btlight.ml.splitters.rolling_timeseries_split import PanelTimeSeriesSplit
from sklearn.decomposition import PCA
from btlight.ml.metrics.scoring import ic_score_func


ic_score = make_scorer(ic_score_func, greater_is_better=True)
# make custom scorer


# pipline 1
# pipeline = Pipeline(
#     [
#         ("pca", PCA(n_components=0.95)),
#         (
#             "regressor",
#             XGBRegressor(
#                 objective="reg:squarederror",
#                 random_state=42,
#                 n_estimators=400,
#                 tree_method="hist",
#             ),
#         ),
#     ]
# )

# param_grid = {
#     "regressor__max_depth": [3, 5, 8],
#     "regressor__learning_rate": [0.03, 0.05],
#     "regressor__subsample": [0.7, 0.9],
#     "regressor__n_estimators": [10, 50, 200, 400],
# }

# pipeline 2
pipeline = Pipeline(
    [
        # not really needed since our input is very tamed
        # ("scaler", StandardScaler()),
        # forcing some overfitting here to test pipeline
        # ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        # simple regressor
        # ("pca", PCA(n_components=0.95)),
        ("regressor", Ridge(random_state=42))
    ]
)

param_grid = {
    # "regressor__alpha": [1e-5, 1e-4, 0.001, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7]
    "regressor__alpha": [0.1, 1.0, 10.0, 100.0]
}


# add validation sets for hyperparameter tuning
# tscv = TimeSeriesSplit(n_splits=2)# This will split without date consideration
# kf = KFold(n_splits=5)
# is actually okay in our setup, but nicer if it is different. I also think
# we should have sector splitter etc.

# grid search object
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    # cv=KFold(n_splits=5),
    cv=PanelTimeSeriesSplit(n_splits=3, date_level="DATE"),
    n_jobs=-1,
    # scoring=ic_score,
    scoring="neg_mean_squared_error",
    refit=True,  # likely default, will refit on entire sample once hyper param is found
)

####################################################
# Time Grid for Rolling Split
####################################################

# this can only be done if X and y are aligned
time_grid = X.index.get_level_values("DATE").unique().sort_values()


####################################################
# Train and Test Rolling Split
####################################################
from btlight.ml.splitters.rolling_timeseries_split import ObservationGridRollingSplit

rolling_splitter = ObservationGridRollingSplit(
    observation_dates=time_grid,
    train_window_obs=12 * 2,  # 2 years of training window
    skip_obs_between_train_test=0,
    retrain_stride=1,  # retrain every quarter
)

rolling_splitter.print_splits(X=X)


####################################################
# look at one example / split
# TODO: make this be built in parallel for all!
####################################################
study = False

if study:
    from sklearn.base import clone

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
    plt.semilogx(df[param_name], df["mean_test_mse"], marker="o", linestyle="-")
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Mean CV MSE")
    plt.title("Ridge: CV Loss vs Alpha")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    # why is the standard deviation so high?
    plt.figure(figsize=(8, 5))

    # Semilog-x plot with shaded std region
    plt.semilogx(
        df[param_name], df["mean_test_mse"], marker="o", linestyle="-", label="Mean CV MSE"
    )
    plt.fill_between(
        df[param_name],
        df["mean_test_mse"] - df["std_test_score"],
        df["mean_test_mse"] + df["std_test_score"],
        alpha=0.2,
    )

    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Mean CV MSE")
    plt.title("Ridge: CV Loss vs Alpha")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    print(
        pd.Series(
            model.best_estimator_.named_steps["regressor"].coef_, index=X.columns
        ).sort_values()
    )


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

print("Training Done")
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

print("Shap Values Computed")
logger.info(f"Saved SHAP values to {shap_path}")


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


# inspecting single split / model
# filter one split
split_id = 0
df_split = df_all_shap[df_all_shap["split"] == split_id].drop(columns="split")

# rebuild Explanation
shap_exp = shap.Explanation(
    values=df_split.values,
    base_values=None,  # optional
    feature_names=df_split.columns,
    data=None,  # optional
)

shap.plots.heatmap(shap_exp[:1000])
