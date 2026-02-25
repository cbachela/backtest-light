import joblib
import shap
from pathlib import Path

from btlight.ml.naming.model_name import resolve_target_name
from btlight.ml.io.model_io import model_path, shap_values_path


def get_shaply_values(
    X,
    y,
    train_idx,
    target_asset=None,
):
    """
    Compute SHAP values for a trained model loaded from disk.

    Args:
        X: Feature data to explain.
        y: Target data for path resolution.
        train_idx: Training index used for path resolution.
        shap_explainer_cls: SHAP explainer class to instantiate.
        target_asset: Optional target asset name.

    Returns:
        SHAP values for X computed by the explainer.
    """

    resolved_target = resolve_target_name(y, target_asset)

    path = model_path(X, y, train_idx, target_asset)

    # load the model from disk, will fail if you don't train one
    model = joblib.load(path)

    if not hasattr(model, "best_estimator_"):
        raise ValueError(
            f"Loaded model does not have 'best_estimator_' attribute. Ensure the model was trained and saved correctly at {path}."
        )

    shap_value_path = shap_values_path(X, y, train_idx, target_asset)

    if Path(shap_value_path).exists():
        print(f"SHAP values already computed and saved at {shap_value_path}. Loading from disk...")
        shap_values = joblib.load(shap_value_path)
        return shap_values

    # Extract pipeline components
    # (loaded model has them)
    pipe = model.best_estimator_

    # get the preprocessing pipeline
    # we need the final features not the raw ones
    preprocessing_pipe = pipe[:-1]

    # get the actual model part
    # for example xgb model itself
    actual_model = pipe[-1]

    if len(pipe.steps) == 1:
        # no preprocessing, using raw data
        X_shap = X.loc[train_idx]
    else:
        # transform the features accordingly
        preprocessing_pipe = pipe[:-1]
        X_shap = preprocessing_pipe.transform(X.loc[train_idx])

    # this is numeric approach that does not include the
    # model nature - makes it weaker but more generalizable to any model type
    explainer = shap.Explainer(actual_model.predict, X_shap)

    # can take a substantial amount of time due to the
    # many perturbations
    shap_values = explainer(X_shap)

    # save the computed SHAP values to disk for future use
    joblib.dump(shap_values, shap_value_path)
    print(f"SHAP values computed and saved at {shap_value_path}.")

    return shap_values
