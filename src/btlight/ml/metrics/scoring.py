import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def cross_sectional_ic(y_true, y_pred):
    df = pd.DataFrame({"y": y_true, "pred": y_pred})
    if isinstance(y_true.index, pd.MultiIndex):
        df.index = y_true.index
        date_level = 0
    else:
        raise ValueError("Need MultiIndex with DATE level")

    ic_list = []
    for _, group in df.groupby(level=date_level):
        if len(group) > 5:
            ic = group["y"].corr(group["pred"], method="spearman")
            if not np.isnan(ic):
                ic_list.append(ic)
    return np.mean(ic_list) if ic_list else 0.0


def ic_score_func(y_true, y_pred):
    return cross_sectional_ic(y_true, pd.Series(y_pred, index=y_true.index))