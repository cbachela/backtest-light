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


def membership_exits(panel):
    """Monthly convention: companies leaving the universe are sold at the rebalance close."""
    return {
        t: set(panel.asset_index[panel.day(t - 1)]) - set(panel.asset_index[panel.day(t)])
        for t in range(1, len(panel.dates))
    }


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


def rolling_splits(
    panel, first_test, last_test=None, train_months=39, validation_months=9, test_months=3
):
    """Walk-forward splits; purge labels that end after a boundary.

    Defaults are Wang et al. (Section 4.1, 39/9/3). train_months=None expands training
    from the first panel date, as in Gu, Kelly and Xiu (2020) for monthly data.
    """
    start = pd.Timestamp(first_test)
    if start.day != 1:
        raise ValueError("first_test must be the first day of a month")
    if (train_months is not None and train_months < 1) or min(validation_months, test_months) < 1:
        raise ValueError("window lengths must be positive")
    end = pd.Timestamp(last_test) if last_test else pd.Timestamp(panel.dates[-1])
    while start <= end:
        val_start = np.datetime64(start - pd.DateOffset(months=validation_months), "D")
        train_start = (
            panel.dates[0]
            if train_months is None
            else np.datetime64(start - pd.DateOffset(months=train_months + validation_months), "D")
        )
        test_start = np.datetime64(start, "D")
        test_end = np.datetime64(start + pd.DateOffset(months=test_months), "D")
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
        start += pd.DateOffset(months=test_months)


def experiment_splits(panel, first_test, factors, smoke=False, **windows):
    found = False
    for quarter, train, val, test in rolling_splits(panel, first_test, **windows):
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


def forecast_diagnostics(mean, realized, variance=None):
    """One date: spread of predicted means, rank IC and median predicted volatility."""
    mean, realized = np.asarray(mean), np.asarray(realized)
    stats = {"mu_spread": float(mean.std())}
    if mean.std() > 0 and realized.std() > 0:
        # Rank correlation; ties are rare in continuous forecasts and broken by order.
        ranks = [np.argsort(np.argsort(x)) for x in (mean, realized)]
        stats["ic"] = float(np.corrcoef(*ranks)[0, 1])
    if variance is not None:
        stats["sigma_median"] = float(np.median(np.sqrt(variance)))
    return stats


def portfolio_diagnostics(weights, previous, exit_weight=0.0):
    """One rebalance: positions above 0.1%, concentration, largest weight, one-way turnover."""
    w, previous = np.asarray(weights), np.asarray(previous)
    return {
        "holdings": int((w > 1e-3).sum()),
        "effective_n": float(1 / np.square(w).sum()),
        "max_weight": float(w.max()),
        "turnover": float((np.abs(w - previous).sum() + exit_weight) / 2),
    }


def average(rows, prefix=""):
    """Average per-date diagnostics; keys missing on some dates use the dates that have them."""
    keys = dict.fromkeys(k for row in rows for k in row)
    return {prefix + k: float(np.mean([row[k] for row in rows if k in row])) for k in keys}


def plain(values):
    """NumPy scalars to Python values, so rows can be written as JSON."""
    return {k: v.item() if hasattr(v, "item") else v for k, v in values.items()}


def periods_per_year(panel):
    # Monthly panels rebalance about every 30 days; daily panels about every trading day.
    gap = np.median(np.diff(panel.dates).astype("timedelta64[D]").astype(int))
    return 12 if gap >= 20 else 252


def period_returns(panel, records):
    """Portfolio (net of the run's cost) and equal-weight benchmark return per test date.

    JKP targets are excess returns, so both series and all metrics are excess returns.
    The equal-weight benchmark holds the whole eligible universe without costs.
    """
    dynamic = isinstance(panel, DynamicPanel)
    rows = []
    for t, weights, cost in records:
        realized = panel.targets[panel.day(t)] if dynamic else panel.raw_returns[t]
        rows.append({
            "date": str(panel.dates[t]),
            "label_end": str(panel.label_end[panel.day(t)][0] if dynamic else panel.label_end[t]),
            "portfolio": float(weights @ realized - cost),
            "equal_weight": float(realized.mean()),
        })
    returns = pd.DataFrame(rows)
    if not np.isfinite(returns.portfolio).all() or (returns.portfolio <= -1).any():
        raise ValueError("invalid portfolio returns")
    return returns


def ratio(a, b):
    return a / b if b > 0 else np.nan


def performance(r, periods):
    """Descriptive statistics of one return series; annualized with `periods` per year."""
    r = np.asarray(r, dtype=float)
    wealth = np.r_[1.0, np.cumprod(1 + r)]
    sd = r.std(ddof=1)
    # Downside deviation below zero; zero is the risk-free rate for excess returns.
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    centred = r - r.mean()
    m2 = np.mean(centred**2)
    return {
        "annualized_excess_return": r.mean() * periods,
        "cagr": wealth[-1] ** (periods / len(r)) - 1,
        "annualized_excess_volatility": sd * np.sqrt(periods),
        "sharpe_ratio": ratio(r.mean(), sd) * np.sqrt(periods),
        "sortino_ratio": ratio(r.mean(), downside) * np.sqrt(periods),
        "max_drawdown_excess": (1 - wealth / np.maximum.accumulate(wealth)).max(),
        "skewness": ratio(np.mean(centred**3), m2**1.5),
        "excess_kurtosis": ratio(np.mean(centred**4), m2**2) - 3,
        "hit_rate": np.mean(r > 0),
        "best_period": r.max(),
        "worst_period": r.min(),
    }


def relative(r, benchmark, periods, prefix):
    """Active-return statistics against one benchmark, plus the benchmark's own CAGR."""
    r, b = np.asarray(r, dtype=float), np.asarray(benchmark, dtype=float)
    active = r - b
    tracking = active.std(ddof=1)
    beta = ratio(np.cov(r, b)[0, 1], b.var(ddof=1))
    return {
        f"{prefix}cagr": np.prod(1 + b) ** (periods / len(b)) - 1,
        f"{prefix}active_return": active.mean() * periods,
        f"{prefix}tracking_error": tracking * np.sqrt(periods),
        f"{prefix}information_ratio": ratio(active.mean(), tracking) * np.sqrt(periods),
        f"{prefix}active_tstat": ratio(active.mean(), tracking) * np.sqrt(len(active)),
        f"{prefix}beta": beta,
        f"{prefix}alpha": (r.mean() - beta * b.mean()) * periods,
        # Final wealth relative to the benchmark's, minus one.
        f"{prefix}cumulative_relative": np.prod(1 + r) / np.prod(1 + b) - 1,
        f"{prefix}hit_rate": np.mean(active > 0),
    }


def summarize(panel, records):
    if len(records) < 2:
        raise ValueError("need at least two test observations")
    returns = period_returns(panel, records)
    periods = periods_per_year(panel)
    r = returns.portfolio.to_numpy()
    return pd.DataFrame([{
        "observations": len(r),
        "test_start": returns.date.iloc[0],
        "test_end": returns.label_end.iloc[-1],
        **performance(r, periods),
        **relative(r, returns.equal_weight.to_numpy(), periods, "ew_"),
    }])


def save_results(results, path, source):
    result = pd.concat(results, ignore_index=True).assign(source=source)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(result.to_string(index=False))
    print(f"Saved {path}")
