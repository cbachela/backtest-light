################################
# Get the data
################################
from pathlib import Path
import pandas as pd


repo_root = Path(__file__).resolve().parents[2]
data_path = repo_root / "data" / "real" / "DMEXLME" / "20251231"

# I prepared the parquets such that they have meta data that 
# in pandas will be index cols
# signals / features on ME grid
X = pd.read_parquet(str(data_path / "signals.parquet"))

# return series, is on daily grid (trading days only)
return_series = pd.read_parquet(str(data_path / "return_series.parquet"))

    
    
# ensure that the format is right
def ensure_multiindex(df):
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != ["DATE", "ID"]:
        df = df.reset_index()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.set_index(["DATE", "ID"]).sort_index()
    else:
        df = df.copy()
        df = df.sort_index()
        df = df.set_index(
            pd.MultiIndex.from_arrays(
                [pd.to_datetime(df.index.get_level_values("DATE")), df.index.get_level_values("ID")],
                names=["DATE", "ID"],
            )
        )
    return df

X = ensure_multiindex(X)
return_series = ensure_multiindex(return_series)

################################
# Feature Engineering
################################
# the signals are highly tamed - most is alrady done.
# see signal service for details
# TODO: add static field like sector etc.
# however note that this is highly dangerous,
# you will get models that learn that a stock is good because
# it is in the tech sector, that is hardly a  model that generalizes well,
# so sector could be considered something over which the model
# should be validated / tuned.
# we can also run sector and country specific models to see
# how strong the difference is..

# sort
X = X.sort_index()
# keep only numeric columns (for now)
X = X.select_dtypes(include="number")

# we are adding cross sectional ranks, 
# like this a stock is aware how it stand relative to the others
for col in X.columns:
    X[f"{col}_rank"] = X.groupby("DATE")[col].rank(pct=True)

time_grid = X.index.get_level_values("DATE").unique().sort_values()
    
################################
# Label Creation
################################
# we shift by one unit, this hence depends on the return series frequency
# currently we have monthly data

import pandas as pd
import numpy as np

def compound_returns_on_grid(
    return_series: pd.Series,
    time_grid: pd.DatetimeIndex,
    date_level: str = "DATE",
    id_level: str = "ID",
) -> pd.DataFrame:
    """
    Compounds daily returns between consecutive dates in time_grid.

    Parameters
    ----------
    return_series : pd.Series
        Daily returns indexed by MultiIndex (DATE, ID).
    time_grid : pd.DatetimeIndex
        Sorted dates defining the feature / prediction grid.
    date_level : str
        Name of the date level in the MultiIndex.
    id_level : str
        Name of the asset identifier level.

    Returns
    -------
    pd.DataFrame
        Compounded returns indexed by (DATE, ID),
        where DATE corresponds to the start of each grid period.
    """

    # ensure proper ordering
    time_grid = pd.DatetimeIndex(time_grid).sort_values()
    return_series = return_series.sort_index()

    out = []

    for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
        mask = (
            (return_series.index.get_level_values(date_level) > t0)
            & (return_series.index.get_level_values(date_level) <= t1)
        )

        period_ret = (
            (1.0 + return_series[mask])
            .groupby(level=id_level)
            .prod()
            .sub(1.0)
        )

        period_ret.index = pd.MultiIndex.from_product(
            [[t0], period_ret.index],
            names=[date_level, id_level],
        )

        out.append(period_ret)

    if not out:
        return pd.DataFrame(columns=["return"])

    return pd.concat(out).to_frame("return")


daily_ret = return_series["tot_return_gross_local"].astype(float)

grid_ret = compound_returns_on_grid(
    return_series=daily_ret,
    time_grid=time_grid,
)

grid_ret = ensure_multiindex(grid_ret)

y = grid_ret.groupby(level="ID")["return"].shift(-1).dropna()

# quick check - looks good
time_grid
grid_ret.index.get_level_values(0).unique()


make_cross_sectional = True

def winsorize(x):
    x = x.clip(x.quantile(0.01), x.quantile(0.99))
    return (x - x.mean()) / x.std()

import numpy as np
def make_xsectional_zscore(y):
    # align index ordering
    y = y.sort_index()

    # cross-sectional z-score per DATE
    y_cs = y.groupby(level="DATE").transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # optional but recommended step to clean it
    y_cs = y_cs.replace([np.inf, -np.inf], np.nan).dropna()

    return y_cs

if make_cross_sectional:
    y = y.groupby(level="DATE").transform(winsorize)
    y = make_xsectional_zscore(y)
    # ensure multiindex is still correct after the transformation
    # drop Nans in the labels, these will be at the end of the series since the future
    # is unkown


y = y.dropna()


################################
# DownSampling
################################


# keep only 10 IDs per DATE to downsample
# This is to keep the code fast for testing purposes,
# once happy remove this of course.
downsample = False
if downsample:
    rng = 42
    X = (
        X.groupby(level="DATE", group_keys=False)
        .apply(lambda df: df.sample(n=min(10, len(df)), random_state=rng))
    )



################################
# Alignment of X and y
################################

common_index = X.index.intersection(y.index)
X = X.loc[common_index].sort_index()
y = y.loc[common_index].sort_index()

print("X.shape", X.shape)
print("y.shape", y.shape)


################################
# Define a model / pipeline
################################
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
from btlight.ml.rolling_timeseries_split import PanelTimeSeriesSplit

def ic_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return spearmanr(y, pred)[0]

# add any fit, predict model
# you can add feature selectors, PCA etc. in the pipeline as well
pipeline = Pipeline(
    [
        (
            "regressor",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_estimators=1000,
                tree_method="hist",
            ),
        ),
    ]
)

param_grid = {
    "regressor__max_depth": [3, 5],
    "regressor__learning_rate": [0.03, 0.05],
    "regressor__subsample": [0.7, 0.9],
}

# add validation sets for hyperparameter tuning
#tscv = TimeSeriesSplit(n_splits=3) This will split without date consideration
# is actually okay in our setup, but nicer if it is different. I also think
# we should have sector splitter etc.

# grid search object
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=PanelTimeSeriesSplit(n_splits=3, date_level="DATE"),
    n_jobs=-1,
    # scoring=ic_scorer, # running into pickable issues with parallel setup (to fix)
    scoring="neg_mean_squared_error",
)

# NOTE: expensive fit, will take a while..
# for the full set this will not run on a single cpu
# within a meaningful timeframe, so we will run 
# this in parallel in the train_func below
study_fit = False

if study_fit:
    grid_search.fit(X, y) 

    # show the best model on this previous test
    # this will be the best from the hyperparameter grid above
    best_model = grid_search.best_estimator_


####################################################
# Time Grid for Rolling Split
####################################################

# this can only be done if X and y are aligned
time_grid = X.index.get_level_values("DATE").unique().sort_values()


####################################################
# Train and Test Rolling Split
####################################################
from btlight.ml.rolling_timeseries_split import ObservationGridRollingSplit

rolling_splitter = ObservationGridRollingSplit(
    observation_dates=time_grid,
    train_window_obs=4, # 1 year
    skip_obs_between_train_test=1,
    retrain_stride=1, # every quarter
)

# below we print the periods, good to understand those properly
# and validate that they make sense
for i, (train_idx, test_idx) in enumerate(rolling_splitter.split(X=X), start=1):
    print(f"\n{'=' * 12} [iteration {i:02d}] {'=' * 12}")
    print("training_period")
    print(
        "start <-> end:",
        train_idx.get_level_values("DATE").min().strftime("%Y-%m-%d"),
        "<->",
        train_idx.get_level_values("DATE").max().strftime("%Y-%m-%d"),
    )
    print("number of samples:", len(train_idx))
    train_dates = (
        train_idx.get_level_values("DATE")
        .unique()
        .sort_values()
        .strftime("%Y-%m-%d")
        .tolist()
    )
    print("number of unique dates:", len(train_dates))
    print("dates:", train_dates)

    print("testing_period")
    print(
        "start <-> end:",
        test_idx.get_level_values("DATE").min().strftime("%Y-%m-%d"),
        "<->",
        test_idx.get_level_values("DATE").max().strftime("%Y-%m-%d"),
    )
    print("number of samples:", len(test_idx))
    test_dates = (
        test_idx.get_level_values("DATE")
        .unique()
        .sort_values()
        .strftime("%Y-%m-%d")
        .tolist()
    )
    print("number of unique dates:", len(test_dates))
    print("dates:", test_dates)



####################################################
# Train (will persist the models to disk)
####################################################
from btlight.ml.traintest import train_func
from sklearn.base import clone
import joblib

#n_jobs = joblib.effective_n_jobs()
n_jobs = 50

retrain = True
if retrain:
    # run the training of the models in parallel, this will take time
    with joblib.parallel_backend(
        "loky", n_jobs=n_jobs, verbose=True
    ):
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
                #target_asset=target_asset,
            )
            for train_idx, _ in rolling_splitter.split(X=X)
        )

        joblib.Parallel()(jobs)

print("Training Done")



####################################################
# Test (will load the models from disk and generate predictions)
####################################################
import joblib
from btlight.ml.traintest import test_func

#n_jobs = joblib.effective_n_jobs()
n_jobs = 50

# run the test, which is simply generating the predictions
# here also in parallel
with joblib.parallel_backend(
    #"loky", n_jobs=joblib.effective_n_jobs(), verbose=True
    "loky", n_jobs=n_jobs, verbose=True
):
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

# build dataframe from y_pred_test (second element of tuple)
# then unstack to have a dataframe with index DATE and columns ID, values are predictions
y_pred_test_df = pd.concat([item[1] for item in result],axis=0,).unstack()

print(y_pred_test_df.head())



####################################################
# Next
####################################################
# - optionally persist the predictions 
# - start with a signal level backtest
# - determine the metric to evalutete performance of the predictions. 
#   Contemplate moving beyond MSE and update also the loss function in 
#   the model to reflect the metric of interest (e.g. Sharpe ratio, Sortino ratio, etc.)
# - Display rolling feature importance, is there persistance and does it make sense
# - Use these ML views to generate quintile portfolios
# - use these views to update prior beliefs in BL approach