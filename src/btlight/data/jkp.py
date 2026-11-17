"""Monthly JKP panel: month-end signals, next-month USD excess return targets."""

import numpy as np
import pandas as pd
from scipy.special import ndtri

from btlight.data.panel import DynamicPanel

# JKP signal names; definitions follow JKP's characteristic documentation.
FEATURES = (
    # Market
    "ret_1_0",
    "ret_3_1",
    "ret_12_1",
    "seas_1_1an",
    "rvol_21d",
    "rmax1_21d",
    "rskew_21d",
    "beta_60m",
    "dolvol_126d",
    "market_equity",
    # Value
    "be_me",
    "sale_me",
    "ocf_me",
    "eqnpo_12m",
    # Profitability and quality
    "op_at",
    "gp_at",
    "ocf_at",
    "ni_be",
    "qmj_prof",
    "qmj_safety",
    # Balance sheet, investment and growth
    "cash_at",
    "debt_at",
    "at_be",
    "dbnetis_at",
    "at_gr1",
    "sale_gr1",
    "ocf_at_chg1",
)
ISSUE_FLAGS = ("obs_main", "exch_main", "common", "primary_sec")


def read_jkp(path, start, end):
    """Read only the needed columns; the asset ID is JKP's security id."""
    columns = ["eom", "id", "excntry", *ISSUE_FLAGS, "me", "rvol_252d", "ret_exc_lead1m", *FEATURES]
    jkp = pd.read_parquet(path, columns=columns).reset_index(drop=True)
    jkp["eom"] = pd.to_datetime(jkp.eom)
    return jkp.loc[jkp.eom.between(pd.Timestamp(start), pd.Timestamp(end))]


def size_screen(rows, fraction=0.2):
    """True for the smallest floor(fraction * n) companies by me on each date; ties by id."""
    if not 0 <= fraction < 1:
        raise ValueError("size fraction must be in [0, 1)")
    order = rows.sort_values(["eom", "me", "id"], kind="stable")
    groups = order.groupby("eom")
    small = groups.cumcount() < np.floor(groups.id.transform("size") * fraction)
    return small.reindex(rows.index)


def prepare_jkp(jkp, size_fraction=0.2):
    """Build the monthly panel.

    Holdings are formed at each month-end close and earn JKP's next-month return.
    Entry and label dates are nominal: the day after each month-end.
    """
    rows = jkp.loc[jkp[list(ISSUE_FLAGS)].eq(1).all(axis=1)].copy()
    rows["id"] = rows.id.astype("int64").astype(str)
    if rows.duplicated(["id", "eom"]).any():
        raise ValueError("duplicate JKP id/eom")
    # Size and risk must be known at the month-end; screen before looking at targets.
    known = np.isfinite(rows.me) & rows.me.gt(0) & np.isfinite(rows.rvol_252d)
    rows = rows.loc[known]
    screened = size_screen(rows, size_fraction)
    size_screened_rows = int(screened.sum())
    rows = rows.loc[~screened]
    # User decision: drop rows without a next-month return (mostly delistings).
    # This uses future availability for about 0.4% of rows; documented look-ahead.
    missing_target = ~np.isfinite(rows.ret_exc_lead1m)
    rows = rows.loc[~missing_target].copy()
    # Excess returns can fall just below -100% (total loss minus the risk-free rate).
    capped_targets = int(rows.ret_exc_lead1m.lt(-1).sum())
    rows["target"] = rows.ret_exc_lead1m.clip(lower=-1)
    rows = rows.loc[rows.groupby("eom").id.transform("size").ge(10)]
    rows = rows.sort_values(["eom", "id"]).reset_index(drop=True)
    if rows.empty:
        raise ValueError("no month-ends with ten eligible companies")

    names = []
    for feature in FEATURES:
        values = rows[feature].where(np.isfinite(rows[feature]))
        ranks = values.groupby(rows.eom).rank(method="average")
        counts = values.groupby(rows.eom).transform("count")
        rows[feature + "_normal"] = ndtri((ranks - 0.5) / counts).fillna(0.0)
        rows[feature + "_missing"] = values.isna()
        names.append(feature + "_normal")
    names += [f + "_missing" for f in FEATURES]

    ids, assets = pd.factorize(rows.id, sort=True)
    counts = rows.groupby("eom", sort=True).size()
    day = pd.Timedelta(days=1)
    panel = DynamicPanel(
        dates=counts.index.to_numpy(dtype="datetime64[D]"),
        offsets=np.r_[0, counts.to_numpy().cumsum()],
        assets=assets.to_numpy(dtype=str),
        asset_index=ids.astype(np.int32),
        features=rows[names].to_numpy(dtype=np.float32),
        feature_names=np.array(names),
        targets=rows.target.to_numpy(dtype=np.float64),
        entry_dates=(rows.eom + day).to_numpy(dtype="datetime64[D]"),
        label_end=(rows.eom + pd.offsets.MonthEnd(1) + day).to_numpy(dtype="datetime64[D]"),
        # rvol_252d is a daily standard deviation; about 21 trading days per month.
        risk_variance=np.maximum(rows.rvol_252d.to_numpy() ** 2 * 21, 1e-8),
    )
    audit = rows[["eom", "id", "excntry", "me", "ret_exc_lead1m", "target"]]
    summary = dict(
        source="JKP only",
        frequency="monthly",
        target="ret_exc_lead1m (USD excess return over the next month), capped at -100%",
        universe=(
            "obs_main/exch_main/common/primary_sec = 1; positive me and finite rvol_252d; "
            f"drop the smallest floor({size_fraction:g} x count) by me; "
            "drop rows without a next-month return; at least 10 companies per month"
        ),
        features="date-wise inverse-normal ranks; missing scores zero with flags",
        risk_variance="rvol_252d^2 x 21",
        feature_names=names,
        size_fraction=size_fraction,
        size_screened_rows=size_screened_rows,
        dropped_missing_targets=int(missing_target.sum()),
        capped_targets=capped_targets,
        rows=len(rows),
        dates=len(panel.dates),
        first_date=str(panel.dates[0]),
        last_date=str(panel.dates[-1]),
        assets=len(panel.assets),
        countries=int(rows.excntry.nunique()),
    )
    try:
        panel.validate_portfolio_dates(range(len(panel.dates)))
        summary["portfolio_dates_valid"] = True
    except ValueError as error:
        summary["portfolio_dates_valid"] = False
        summary["first_portfolio_error"] = str(error)
    return panel, audit, summary
