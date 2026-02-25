import pandas as pd
from btlight.ml.naming.model_name import resolve_start_end_dates_from_index, resolve_target_name


def model_path(X, y, train_idx, target_asset=None) -> str:

    assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
    assert isinstance(y, (pd.DataFrame, pd.Series)), "y must be a DataFrame or Series"

    # resolve target name
    resolved_target = resolve_target_name(y, target_asset)

    # start and end date of the training period (handle MultiIndex/tuple-index)
    start_ts, end_ts = resolve_start_end_dates_from_index(train_idx)
    start_dt, end_dt = start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")

    path = f"/tmp/{resolved_target}_{start_dt}_{end_dt}_model.pkl"
    return path


def shap_values_path(X, y, train_idx, target_asset=None) -> str:

    model_pkl_path = model_path(X, y, train_idx, target_asset)
    shap_value_path = model_pkl_path.replace("_model.pkl", "_shap_values.pkl")
    return shap_value_path
