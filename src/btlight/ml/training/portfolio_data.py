"""Risk estimates, purged splits and reporting shared by the thesis models."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from btlight.data.panel import DynamicPanel


def align_holdings(previous_ids, previous_weights, current_ids, executed_exits=()):
    """Align long-only holdings by ID, retaining gradients through weight changes.

    A missing security is not a sale. The execution ledger must explicitly confirm
    exits before their weight can be removed. Zeros here mean no holdings, not returns.
    """
    previous_ids, current_ids = list(previous_ids), list(current_ids)
    if len(set(previous_ids)) != len(previous_ids) or len(set(current_ids)) != len(
        current_ids
    ):
        raise ValueError("duplicate security IDs")
    if previous_weights.shape != (len(previous_ids),):
        raise ValueError("holding shape mismatch")
    lookup = {asset: i for i, asset in enumerate(previous_ids)}
    current, sold = set(current_ids), set(executed_exits)
    if current & sold:
        raise ValueError(
            "an executed forced exit cannot also be allocated at this rebalance"
        )
    removed = [i for i, asset in enumerate(previous_ids) if asset not in current]
    unresolved = [
        previous_ids[i]
        for i in removed
        if previous_ids[i] not in sold
        and abs(previous_weights[i].detach().item()) > 1e-8
    ]
    if unresolved:
        raise ValueError(
            f"Unresolved holdings need an executable exit or retention: {unresolved[:5]}"
        )
    zero = previous_weights.new_zeros(())
    aligned = torch.stack(
        [previous_weights[lookup[a]] if a in lookup else zero for a in current_ids]
    )
    exited = previous_weights[removed].abs().sum() if removed else zero
    return aligned, exited


def exit_lookup(panel, path):
    """Match audited exits to the portfolio's actual rebalance close.

    Accept aligned sales and explicit terminal distributions already booked in
    panel returns. Terminal settlements can include labelled recovery assumptions.
    Other delayed sales still require a pending order/valuation ledger.
    """
    events = pd.read_parquet(path)
    if "asset" in events and "exit_price" in events:
        allowed_reasons = {"daily_ineligible"}
    else:
        if not {"gvkey", "iid"}.issubset(events):
            raise ValueError("exit audit needs asset IDs or gvkey/iid")
        events["asset"] = events.gvkey.astype(str) + ":" + events.iid.astype(str)
        events = events.rename(columns={"exit_price_eur": "exit_price"})
        allowed_reasons = {"bottom_20pct"}
    required = {
        "asset",
        "decision_date",
        "execution_date",
        "reason",
        "status",
        "exit_price",
    }
    if not required.issubset(events):
        raise ValueError("invalid exit audit")
    if events.empty:
        return {}
    valid = events.reason.isin(allowed_reasons) & events.status.eq("price_available")
    valid &= np.isfinite(events.exit_price) & events.exit_price.gt(0)
    if "event_date" in events:
        settled = events.reason.eq("corporate_action") & events.status.eq("terminal_settled")
        settled &= events.event_date.notna() & events.event_date.le(events.execution_date)
        settled &= np.isfinite(events.exit_price) & events.exit_price.ge(0)
        valid |= settled
    valid &= events.execution_date.gt(events.decision_date)
    if "scheduled_date" in events:
        # Delayed orders must not be booked as immediate exits from this decision.
        valid &= events.execution_date.eq(events.scheduled_date)
    events = events.loc[valid].copy()
    lookup = {asset: i for i, asset in enumerate(panel.assets)}
    events["asset_index"] = events.asset.map(lookup)
    groups = {
        pd.Timestamp(date): group for date, group in events.groupby("decision_date")
    }
    result = {}
    for t, date in enumerate(panel.dates):
        group = groups.get(pd.Timestamp(date))
        if group is None:
            continue
        sl = panel.day(t)
        entries = np.unique(panel.entry_dates[sl])
        if len(entries) == 1:
            result[t] = set(
                group.loc[
                    group.execution_date.eq(pd.Timestamp(entries[0])), "asset_index"
                ]
                .dropna()
                .astype(int)
            )
    return result


def covariance_factors(panel, lookback=252, floor=1e-8):
    """Wang Section 4.4: one-year diagonal risk, using only known returns."""
    if lookback < 2 or floor <= 0:
        raise ValueError("lookback >= 2 and positive variance floor required")
    result = {}
    for t in range(lookback - 1, len(panel.dates)):
        variance = panel.risk_returns[t - lookback + 1 : t + 1].var(axis=0, ddof=1)
        result[t] = torch.diag(
            torch.tensor(np.maximum(variance, floor) ** 0.5, dtype=torch.float64)
        )
    return result


def correlation_factors(panel, lookback=252, shrinkage=0.1):
    """Rolling return correlations, shrunk toward identity; no future labels used.

    This historical correlation proxy is not a fitted DCC or residual model.
    """
    if lookback < 2 or not 0 < shrinkage <= 1:
        raise ValueError("lookback >= 2 and shrinkage in (0, 1] required")
    result = {}
    n = len(panel.assets)
    for t in range(lookback - 1, len(panel.dates)):
        history = panel.risk_returns[t - lookback + 1 : t + 1]
        centered = history - history.mean(axis=0)
        norm = np.sqrt((centered**2).sum(axis=0))
        standardized = centered / np.where(norm > 0, norm, 1)
        correlation = standardized.T @ standardized
        # Constant histories carry no measured correlation, but unit diagonal.
        np.fill_diagonal(correlation, 1.0)
        correlation = (1 - shrinkage) * correlation + shrinkage * np.eye(n)
        result[t] = torch.linalg.cholesky(
            torch.tensor(correlation, dtype=torch.float64)
        )
    return result


def rolling_splits(panel, first_test, last_test=None):
    """Wang et al., Section 4.1: 39/9/3 months; purge unavailable labels."""
    start = pd.Timestamp(first_test)
    if start.day != 1 or start.month not in (1, 4, 7, 10):
        raise ValueError("first_test must be a calendar quarter start")
    end = pd.Timestamp(last_test) if last_test else pd.Timestamp(panel.dates[-1])
    while start <= end:
        train_start = np.datetime64(start - pd.DateOffset(months=48), "D")
        val_start = np.datetime64(start - pd.DateOffset(months=9), "D")
        test_start = np.datetime64(start, "D")
        test_end = np.datetime64(start + pd.DateOffset(months=3), "D")
        d = panel.dates
        if isinstance(panel, DynamicPanel):
            # Fail before purging; unknown outcomes must not silently remove dates.
            panel.validate_portfolio_dates(
                np.flatnonzero((d >= train_start) & (d < test_end))
            )
        label = (
            np.array([panel.label_end[panel.day(t)].max() for t in range(len(d))])
            if isinstance(panel, DynamicPanel)
            else panel.label_end
        )
        train = np.flatnonzero(
            (d >= train_start) & (d < val_start) & (label < val_start)
        )
        val = np.flatnonzero((d >= val_start) & (d < test_start) & (label < test_start))
        test = np.flatnonzero((d >= test_start) & (d < test_end))
        if all(len(x) for x in (train, val, test)):
            yield str(start.date()), train, val, test
        start += pd.DateOffset(months=3)


def experiment_splits(panel, first_test, factors, smoke=False):
    found = False
    for quarter, train, val, test in rolling_splits(panel, first_test):
        train, val, test = [
            np.array([t for t in ids if t in factors]) for ids in (train, val, test)
        ]
        if smoke:
            train, val, test = train[-16:], val[-8:], test[:8]
        if not all(len(ids) for ids in (train, val, test)):
            raise ValueError("not enough history for the requested split")
        found = True
        yield quarter, train, val, test
        if smoke:
            break
    if not found:
        raise ValueError("no usable splits; check first-test and panel dates")


def summarize(panel, records):
    if len(records) < 2:
        raise ValueError("need at least two test observations")
    dynamic = isinstance(panel, DynamicPanel)
    raw = np.array(
        [
            weights @ (panel.targets[panel.day(t)] if dynamic else panel.raw_returns[t])
            - cost
            for t, weights, cost in records
        ]
    )
    excess = (
        raw
        if dynamic
        else np.array(
            [weights @ panel.targets[t] - cost for t, weights, cost in records]
        )
    )
    if not np.isfinite(raw).all() or np.any(raw <= -1):
        raise ValueError("invalid portfolio returns")
    wealth = np.r_[1.0, np.cumprod(1 + raw)]
    return pd.DataFrame(
        [
            {
                "observations": len(raw),
                "annualized_return": raw.mean() * 252,
                "annualized_excess_return": excess.mean() * 252,
                "annualized_volatility": raw.std(ddof=1) * np.sqrt(252),
                "test_start": str(panel.dates[records[0][0]]),
                "test_end": str(
                    panel.label_end[panel.day(records[-1][0])][0]
                    if dynamic
                    else panel.label_end[records[-1][0]]
                ),
                "information_ratio": (
                    excess.mean() / excess.std(ddof=1) * np.sqrt(252)
                    if excess.std(ddof=1) > 0
                    else np.nan
                ),
                "max_drawdown": (1 - wealth / np.maximum.accumulate(wealth)).max(),
            }
        ]
    ).rename(
        columns=(
            {
                "annualized_excess_return": "annualized_target_return",
                "information_ratio": "raw_return_to_volatility",
            }
            if dynamic
            else {}
        )
    )


def save_results(results, path, source):
    result = pd.concat(results, ignore_index=True).assign(source=source)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(result.to_string(index=False))
    print(f"Saved {path}")
