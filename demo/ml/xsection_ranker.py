import sys
sys.path.insert(0, '/Users/matej_ofenhgz2/Desktop/MasterThesis_repository/backtest-light/src')


################################
# Control Params
################################

label_recompute = True
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
data_path = Path("/Users/matej_ofenhgz2/Desktop/MasterThesis_repository/backtest-light/data")
signal_path = data_path / "signals.csv"
return_series_path = data_path / "return_series.txt"

# generated input to training
feature_path = str(data_path / "features.parquet")
label_path = str(data_path / "labels.parquet")

# generated predictions
prediction_path =str(data_path / "ml_signal.parquet")
shap_path = str(data_path / "shap_values.parquet")


################################
# Features
################################
import pandas as pd
from btlight.ml.utils.format import check_if_multiindex, ensure_datetime_index

# only True for testing purposes to make the code fast
downsample = False

if Path(feature_path).exists() and not feature_recompute:

    X = pd.read_parquet(feature_path)

else:

    # load signals
    X = pd.read_csv(signal_path)

    X = X.rename(columns={"date": "DATE"})
    X = X.set_index(["DATE", "ID"])

    # inconsistent naming in signal service with lowercase forces us to
    # todo: this shouldb't happen blindly
    # X.index.names = ["DATE", "ID"]
    print(X.columns.tolist())
    print(X.head())


    # check if we have the proper panel format
    check_if_multiindex(X)

    # check if date level is datetime, if not convert
    X = ensure_datetime_index(X)

    # sort
    X = X.sort_index()

    # keep only numeric columns (for now)
    X = X.select_dtypes(include="number")

    # in a first shot allow this drastic feature reduction
    #selection = ["profitability", "growth_qa", "momentum", "value_sector_stdz", "volatility", "safety", "investment"]
    #selection = ["profitability", "value"]
    #X = X[selection]
    
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
    return_series = pd.read_csv(return_series_path, sep = ',', parse_dates= ["DATE"], usecols=["DATE", "ID", "tot_return_gross"])
    #return_series = pd.read_parquet(return_series_path)
    return_series = return_series.set_index(["DATE", "ID"])
    print(return_series.head())
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

    # 2. Market return per date = equalweighted 
    market_ret = y.groupby(level="DATE").transform("mean")

    y_adj = y - market_ret

    # label transformation pipeline
    label_pipeline = TransformPipeline(
        [
            #CrossSectionalZScore(),
            CrossSectionalWinsorize(lower=0.01, upper=0.99), 
        ]
    )

    y = label_pipeline.fit_transform(y_adj)
    y = y.dropna()

    y = y.groupby(level="DATE").rank(
    method    = 'first',
    ascending = True,
    ).astype(int)

    y = (100 * y / y.groupby(level="DATE").transform("count")).astype(int)


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
from btlight.ml.models.xgb_ranker_wrapper import XGBRankerSklearnWrapper
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer, ndcg_score
from btlight.ml.splitters.rolling_timeseries_split import PanelTimeSeriesSplit
from sklearn.decomposition import PCA
from btlight.ml.metrics.scoring import ic_score_func
from btlight.ml.metrics.scoring import ndcg_scorer

# make custom scorer
# ic_score = make_scorer(ic_score_func, greater_is_better=True)
ltr_scorer = make_scorer(ndcg_scorer, greater_is_better=True)

# Pipeline with XGBRanker — analog to Ridge pipeline
# No preprocessing needed (signals already normalised)
pipeline = Pipeline([
    ("ranker", XGBRankerSklearnWrapper(
        objective   = "rank:ndcg",
    ))
])
 
# Hyperparameter grid — analog to Ridge param_grid
param_grid = {
    "ranker__max_depth"      : [3, 5],
    "ranker__learning_rate"  : [0.01, 0.1],
    "ranker__n_estimators"   : [50, 150, 300],
    "ranker__min_child_weight": [5, 15],
    "ranker__subsample"      : [0.7, 0.9],
}


# GridSearchCV — analog to Ridge
# PanelTimeSeriesSplit respects the temporal structure of panel data
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv      = PanelTimeSeriesSplit(n_splits=3, date_level="DATE"),
    scoring = ltr_scorer,
    n_jobs  = -1,
    refit   = True,
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
from btlight.ml.splitters.rolling_timeseries_split import ExpandingWindowSplit

#rolling_splitter = ExpandingWindowSplit(
 #   min_train_obs  = 36,   # mind. 3 Jahre bevor erster Test-Split
  #  retrain_stride = 1,    # jeden Monat neu testen
#)

rolling_splitter = ObservationGridRollingSplit(
    observation_dates           = time_grid,
    train_window_obs            = 12 * 3,
    skip_obs_between_train_test = 0,
    retrain_stride              = 1,
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
    param_name = "alpha" # or alpha or what is suitable
    df = df.sort_values(by=[param_name], ascending=False)
    #df = df.sort_values(by=["alpha"], ascending=False)
    
    
    # simple Hyperparam plot for Ridge Regression
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.semilogx(df[param_name], df['mean_test_mse'], marker='o', linestyle='-')
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Mean CV MSE")
    plt.title("Ridge: CV Loss vs Alpha")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()
    
    # why is the standard deviation so high?
    plt.figure(figsize=(8,5))
    
    # Semilog-x plot with shaded std region
    plt.semilogx(df[param_name], df['mean_test_mse'], marker='o', linestyle='-', label='Mean CV MSE')
    plt.fill_between(
        df[param_name],
        df['mean_test_mse'] - df['std_test_score'],
        df['mean_test_mse'] + df['std_test_score'],
        alpha=0.2
    )
    
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Mean CV MSE")
    plt.title("Ridge: CV Loss vs Alpha")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


    print(pd.Series(model.best_estimator_.named_steps["regressor"].coef_, index=X.columns).sort_values())
    
    

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


####################################################
# Feature Importance
####################################################
from btlight.ml.io.model_io import model_path
import xgboost as xgb
 
last_split     = list(rolling_splitter.split(X=X))[-1]
last_train_idx = last_split[0]
last_path      = model_path(X, y, last_train_idx, target_asset=None)
last_model     = joblib.load(last_path)
 
# Extract XGBRanker from pipeline
ranker = last_model.best_estimator_.named_steps["ranker"]
 
xgb.plot_importance(
    ranker.get_booster(),
    importance_type  = 'gain',
    max_num_features = 20,
    title            = 'Feature Importance (gain)',
)
plt.tight_layout()
plt.savefig(str(data_path / 'feature_importance_ltr.png'), dpi=150)
plt.show()



# cd /Users/matej_ofenhgz2/Desktop/MasterThesis_repository/backtest-light
# uv run python examples/ml/xsection_ranker.py


####################################################
# NDCG Evaluation — Ist das Modell gut?
####################################################
from sklearn.metrics import ndcg_score
import numpy as np
from btlight.ml.metrics.scoring import ndcg_scorer

print("\n" + "="*50)
print("NDCG EVALUATION")
print("="*50)

ndcg_scores = []

for train_idx, test_idx in rolling_splitter.split(X=X):

    # Predictions für diesen Split
    X_test  = X.loc[test_idx]
    y_test  = y.loc[test_idx]

    # Modell laden
    from btlight.ml.io.model_io import model_path
    path  = model_path(X, y, train_idx, target_asset=None)
    model = joblib.load(path)

    y_pred = model.predict(X_test)
    score  = ndcg_scorer(y_true=y.loc[test_idx], y_pred=y_pred)
    ndcg_scores.append(score)

# Resultate ausgeben
ndcg_series = pd.Series(ndcg_scores)
print(f"Durchschnittlicher NDCG:  {ndcg_series.mean():.4f}")
print(f"Bester NDCG:              {ndcg_series.max():.4f}")
print(f"Schlechtester NDCG:       {ndcg_series.min():.4f}")
print(f"Std NDCG:                 {ndcg_series.std():.4f}")
print()
print("Interpretation:")
print(f"  NDCG = 0.5 → Zufall")
print(f"  NDCG = 1.0 → Perfekt")
print(f"  Dein NDCG  → {ndcg_series.mean():.4f}")

if ndcg_series.mean() > 0.55:
    print("  → Modell ist besser als Zufall ✅")
elif ndcg_series.mean() > 0.50:
    print("  → Modell ist leicht besser als Zufall ⚠️")
else:
    print("  → Modell ist nicht besser als Zufall ❌")

# Plot
ndcg_series.plot(
    title   = 'NDCG Score über Zeit',
    figsize = (10, 4),
    grid    = True,
)
plt.axhline(y=0.5, color='r', linestyle='--', label='Zufall (0.5)')
plt.ylabel('NDCG Score')
plt.xlabel('Split')
plt.legend()
plt.tight_layout()
plt.savefig(str(data_path / 'ndcg_over_time.png'), dpi=150)
plt.close()

print(f"\nNDCG Plot gespeichert!")