import os
import joblib
import pandas as pd


def _resolve_start_end_dates_from_index(idx):
    """Return (start_dt, end_dt) Timestamps extracted from the first level of `idx`.

    Handles:
    - DatetimeIndex
    - MultiIndex (uses level 0)
    - Index of tuple-labels where first element is a date-like value
    """
    if idx is None or len(idx) == 0:
        raise ValueError("index is empty")

    if isinstance(idx, pd.MultiIndex):
        dates = pd.DatetimeIndex(idx.get_level_values(0))
    else:
        first = idx[0]
        if isinstance(first, tuple):
            # assume first element of tuple is the date level
            try:
                dates = pd.DatetimeIndex([t[0] for t in idx])
            except Exception:
                dates = pd.DatetimeIndex(idx)
        else:
            dates = pd.DatetimeIndex(idx)

    dates = dates.sort_values()
    return dates[0], dates[-1]


def train_func(model, X, y, train_idx, target_asset=None):
    """
    Trains a model on the specified target asset and saves the trained model to disk.

    Parameters:
    ----------
    model : object
        The machine learning model to be trained. It should have a `.fit()` method and a `.predict()` method.
    X : pd.DataFrame
        A DataFrame containing the features (independent variables) for training the model.
    y : pd.DataFrame or pd.Series
        The target labels. If a `pd.DataFrame` is supplied, `target_asset` must be provided.
    train_idx : pd.Index
        The index corresponding to the training period.
    target_asset : str or None
        The asset (column) in `y` that the model will predict. If `y` is a
        `pd.Series`, this parameter is optional; when omitted the function will
        use `y.name` or the fallback string `'target'` as the asset name.
    model : object
        The machine learning model to be trained. It should have a `.fit()` method and a `.predict()` method.
    X : pd.DataFrame
        A DataFrame containing the features (independent variables) for training the model.
    y : pd.DataFrame
        A DataFrame containing the target labels (dependent variables) for training. The target asset should be a column in `y`.
    train_idx : pd.Index
        The index corresponding to the training period. It defines which rows of `X` and `y` will be used for training.

    Returns:
    -------
    None
        This function trains the model and saves it to disk, but does not return any value.

    """

    assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
    assert isinstance(y, (pd.DataFrame, pd.Series)), "y must be a DataFrame or Series"

    # resolve target name early so filenames and checks are deterministic
    resolved_target = target_asset
    if isinstance(y, pd.Series):
        if resolved_target is None:
            resolved_target = y.name or "target"
    else:
        # y is a DataFrame: require explicit target_asset
        assert resolved_target is not None, "target_asset must be provided when y is a DataFrame"
        assert resolved_target in y.columns, f"no label found for {resolved_target}"

    # get start and end date of training index (handle MultiIndex/tuple-index)
    start_ts, end_ts = _resolve_start_end_dates_from_index(train_idx)
    start_dt, end_dt = start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")

    # load from disk if exists already
    if os.path.exists(f"/tmp/{resolved_target}_{start_dt}_{end_dt}_model.pkl"):
        return

    # restrict features and labels
    X_train = X.loc[train_idx]
    if isinstance(y, pd.Series):
        y_train = y.loc[train_idx]
    else:
        y_train = y.loc[train_idx][resolved_target]

    assert len(X_train) == len(y_train), (len(X_train), len(y_train))

    # fit the model
    model.fit(X=X_train, y=y_train)

    # generate predictions in training set (optional - kept for parity)
    y_pred = pd.DataFrame(model.predict(X=X_train), index=X_train.index, columns=["y_pred"])

    joblib.dump(model, f"/tmp/{resolved_target}_{start_dt}_{end_dt}_model.pkl")


def test_func(X, y, train_idx, test_idx, target_asset=None):
    """
    Tests a trained model on the specified target asset and returns the predictions.

    Parameters:
    ----------
    X : pd.DataFrame
        A DataFrame containing the features (independent variables) for making predictions.
    y : pd.DataFrame or pd.Series
        The target labels. If a `pd.DataFrame` is supplied, `target_asset` must be provided.
    train_idx : pd.Index
        The index corresponding to the training period used to identify the trained model.
    test_idx : pd.Index
        The index corresponding to the test period used for predictions.
    target_asset : str or None
        The asset (column) in `y` that the model will predict. If `y` is a
        `pd.Series`, this parameter is optional; when omitted the function will
        use `y.name` or the fallback string `'target'` as the asset name.
    X : pd.DataFrame
        A DataFrame containing the features (independent variables) for making predictions.
    y : pd.DataFrame
        A DataFrame containing the target labels (dependent variables). The target asset should be a column in `y`.
    train_idx : pd.Index
        The index corresponding to the training period. It defines which rows of `X` and `y` were used to train the model.
    test_idx : pd.Index
        The index corresponding to the test period. It defines which rows of `X` and `y` are used for testing the model.

    Returns:
    -------
    tuple
        Always returns a 3-tuple:
            - `resolved_target` : str, the name of the target asset.
            - `y_pred_test` : pd.DataFrame, predicted values for `test_idx`.
            - `y_pred_train` : pd.DataFrame, in-sample predicted values for `train_idx`.
    """

    assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
    assert isinstance(y, (pd.DataFrame, pd.Series)), "y must be a DataFrame or Series"

    # resolve target name
    resolved_target = target_asset
    if isinstance(y, pd.Series):
        if resolved_target is None:
            resolved_target = y.name or "target"
    else:
        assert resolved_target is not None, "target_asset must be provided when y is a DataFrame"
        assert resolved_target in y.columns, f"no label found for {resolved_target}"

    # start and end date of the training period (handle MultiIndex/tuple-index)
    start_ts, end_ts = _resolve_start_end_dates_from_index(train_idx)
    start_dt, end_dt = start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")

    # load the model from disk, will fail if you don't train one
    model = joblib.load(f"/tmp/{resolved_target}_{start_dt}_{end_dt}_model.pkl")

    # extract the testing part (forecasting)
    # note in live there is no label and we need another function
    X_test = X.loc[test_idx]

    if isinstance(y, pd.Series):
        y_test = y.loc[test_idx]
    else:
        y_test = y.loc[test_idx][resolved_target]

    # prediction dataframe for test set
    y_pred_test = pd.DataFrame(model.predict(X=X_test), index=X_test.index, columns=["y_pred"])

    # also produce in-sample predictions on the training set
    X_train = X.loc[train_idx]
    y_pred_train = pd.DataFrame(model.predict(X=X_train), index=X_train.index, columns=["y_pred"])

    return (resolved_target, y_pred_test, y_pred_train)
