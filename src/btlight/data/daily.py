"""Daily supplied-price returns with availability-aware monthly JKP features."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

from btlight.data.panel import DynamicPanel

MARKET_FEATURES = (
    "return_1",
    "return_5",
    "return_21",
    "momentum_63_21",
    "volatility_21",
    "volatility_63",
    "max_return_21",
)
JKP_FEATURES = (
    "be_me",
    "op_at",
    "gp_at",
    "ocf_at",
    "cash_at",
    "debt_at",
    "at_gr1",
    "sale_gr1",
)
OPTIONAL_FEATURES = ("liquidity", *JKP_FEATURES)


def check_stock_splits(market, security, fx=None):
    """Audit factor changes over supplied quote intervals; never alter prices.

    security requires Compustat gvkey/iid/datadate/prccd/ajexdi columns.
    Both endpoint prices must match a unique issue to propose a return correction.
    Pass GBP-base Compustat FX rows to compare native quotes with supplied EUR prices.
    Factor changes are split candidates, not independently verified split events.
    Pass scheduled market rows to check the same intervals as prepare_daily.
    """
    quotes = market[["company", "date", "price"]].copy()
    quotes["company"] = quotes.company.astype(str).str.zfill(6)
    quotes["date"] = pd.to_datetime(quotes.date)
    if quotes.duplicated(["company", "date"]).any():
        raise ValueError("duplicate market company/date")
    quotes = quotes.sort_values(["company", "date"])
    quotes[["previous_date", "previous_price"]] = quotes.groupby("company")[[
        "date", "price"
    ]].shift()
    ref = security[["gvkey", "iid", "datadate", "prccd", "ajexdi"]].rename(
        columns={"gvkey": "company", "datadate": "date"}
    ).copy()
    ref["company"] = ref.company.astype(str).str.zfill(6)
    ref["date"] = pd.to_datetime(ref.date)
    ref["match_price"] = ref.prccd
    if fx is not None:
        rates = fx.copy()
        rates["datadate"] = pd.to_datetime(rates.datadate)
        rates["exratd"] = pd.to_numeric(rates.exratd)
        if rates.fromcurd.nunique() != 1:
            raise ValueError("FX rates must share one base currency")
        table = rates.pivot(index="datadate", columns="tocurd", values="exratd")
        table[rates.fromcurd.iloc[0]] = 1.0
        eur = table.div(table["EUR"], axis=0).rdiv(1).rename_axis(columns="curcdd")
        keys = pd.MultiIndex.from_arrays([ref.date, security.curcdd])
        ref["match_price"] *= eur.stack().reindex(keys).to_numpy()
    if ref.duplicated(["company", "iid", "date"]).any():
        raise ValueError("duplicate security company/issue/date")
    rows = quotes.merge(ref, on=["company", "date"], how="inner")
    previous = ref.rename(columns={
        "date": "previous_date", "prccd": "previous_prccd",
        "ajexdi": "previous_ajexdi",
        "match_price": "previous_match_price",
    })
    rows = rows.merge(previous, on=["company", "iid", "previous_date"], how="inner")
    numeric = ["price", "previous_price", "prccd", "previous_prccd", "ajexdi", "previous_ajexdi"]
    valid = np.isfinite(rows[numeric]).all(axis=1) & rows[numeric].gt(0).all(axis=1)
    rows = rows.loc[valid].copy()
    rows["price_match"] = np.isclose(rows.price, rows.match_price, rtol=1e-5, atol=1e-8) & np.isclose(
        rows.previous_price, rows.previous_match_price, rtol=1e-5, atol=1e-8
    )
    # Count matches before selecting factor changes: another issue may match too.
    rows["matching_issues"] = rows.groupby(["company", "date"]).price_match.transform("sum")
    rows["factor_ratio"] = rows.ajexdi / rows.previous_ajexdi
    rows = rows.loc[~np.isclose(rows.factor_ratio, 1, rtol=1e-8, atol=0)].copy()
    rows["return_raw"] = rows.price / rows.previous_price - 1
    matched = rows.price_match & rows.matching_issues.eq(1)
    rows["proposed_return"] = ((1 + rows.return_raw) / rows.factor_ratio - 1).where(matched)
    rows["status"] = np.where(matched, "matched_candidate", "unverified_issue")
    return rows.reset_index(drop=True)


def validate_stock_splits(candidates, security):
    """Check issue matches against currency and inverse shares-outstanding changes.

    Allow up to 2% concurrent issuance, but it must explain at least 90% of
    the log factor change. This prevents unchanged shares passing small events.
    Other corporate actions require a separate, documented event decision.
    """
    ref = security[["gvkey", "iid", "datadate", "cshoc", "curcdd"]].rename(
        columns={"gvkey": "company", "datadate": "date"}
    ).copy()
    ref["company"] = ref.company.astype(str).str.zfill(6)
    ref["date"] = pd.to_datetime(ref.date)
    rows = candidates.merge(ref, on=["company", "iid", "date"], how="left", validate="many_to_one")
    rows = rows.merge(ref.rename(columns={
        "date": "previous_date", "cshoc": "previous_cshoc", "curcdd": "previous_curcdd",
    }), on=["company", "iid", "previous_date"], how="left", validate="many_to_one")
    rows["share_check"] = rows.cshoc / rows.previous_cshoc * rows.factor_ratio
    known = np.isfinite(rows.share_check) & rows.share_check.gt(0)
    tolerance = np.minimum(0.02, np.abs(np.log(rows.factor_ratio)) * 0.1)
    consistent = known & np.abs(np.log(rows.share_check.where(known))).le(tolerance)
    currency = rows.curcdd.notna() & rows.curcdd.eq(rows.previous_curcdd)
    material = (rows.factor_ratio - 1).abs().gt(1e-4)
    rows["decision"] = "not_applied"
    rows["evidence"] = "issue prices do not uniquely match"
    matched = rows.status.eq("matched_candidate")
    rows.loc[matched, "evidence"] = "shares change inconsistent with a pure split/distribution"
    rows.loc[matched & ~currency, "evidence"] = "currency changed or missing"
    rows.loc[matched & ~material, "evidence"] = "factor change below 0.01%; rounding-scale change"
    approved = matched & currency & material & consistent
    rows.loc[approved, "decision"] = "apply"
    rows.loc[approved, "evidence"] = "unique endpoint prices; same currency; inverse shares change"
    return rows


def apply_stock_splits(market, decisions):
    """Apply approved events forward, leaving the source DataFrame unchanged.

    Endpoint checks prevent use with another price series or double adjustment.
    Event date is the end of the checked interval, including calendar closures.
    """
    if "source_price" in market:
        raise ValueError("market already has adjusted prices")
    result = market.copy()
    result["company"] = result.company.astype(str).str.zfill(6)
    result["date"] = pd.to_datetime(result.date)
    events = decisions.loc[decisions.decision.eq("apply")].copy()
    events["company"] = events.company.astype(str).str.zfill(6)
    for column in ("date", "previous_date"):
        events[column] = pd.to_datetime(events[column])
    if events.duplicated(["company", "date"]).any():
        raise ValueError("multiple approved adjustments for a company/date")
    if not (np.isfinite(events.factor_ratio) & events.factor_ratio.gt(0)).all():
        raise ValueError("adjustment factors must be positive and finite")
    if not events.previous_date.lt(events.date).all():
        raise ValueError("adjustment interval must have increasing dates")
    prices = result.set_index(["company", "date"]).price
    for date_col, price_col in (("date", "price"), ("previous_date", "previous_price")):
        keys = pd.MultiIndex.from_arrays([events.company, events[date_col]])
        if not np.isclose(prices.reindex(keys), events[price_col], rtol=1e-8, atol=1e-10).all():
            raise ValueError("approved adjustment does not match source endpoint prices")
    factors = events.set_index(["company", "date"]).factor_ratio
    result = result.sort_values(["company", "date"])
    result["split_factor"] = factors.reindex(pd.MultiIndex.from_frame(result[["company", "date"]])).to_numpy()
    result["split_corrected"] = result.split_factor.notna()
    cumulative = result.split_factor.fillna(1).groupby(result.company).cumprod()
    result["source_price"] = result.price
    result["price"] = result.price / cumulative
    return result.drop(columns="split_factor").sort_values(["date", "company"]).reset_index(drop=True)


def trading_calendar(market):
    """Skip dates with country-wide inactivity; retain the evidence for review.

    At least five previously quoted companies and 95% missing quotes or zero
    supplied liquidity indicate a systematic closure, not an isolated gap.
    This is an inferred schedule, not an official exchange holiday calendar.
    """
    price = market.pivot(index="date", columns="company", values="price")
    liquidity = market.pivot(index="date", columns="company", values="liquidity")
    country = market.pivot(index="date", columns="company", values="country").ffill()
    prior = price.shift(1).notna()
    rows = []
    for code in sorted(market.country.unique()):
        active = prior & country.eq(code)
        counts = active.sum(axis=1)
        missing = (active & price.isna()).sum(axis=1)
        inactive = (active & (price.isna() | liquidity.eq(0))).sum(axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "date": price.index,
                    "country": code,
                    "previous_quotes": counts.to_numpy(),
                    "missing_quotes": missing.to_numpy(),
                    "inactive_quotes": inactive.to_numpy(),
                    "no_trade": (counts.ge(5) & inactive.div(counts).ge(0.95)).to_numpy(),
                }
            )
        )
    audit = pd.concat(rows, ignore_index=True)
    closed = pd.Index(audit.loc[audit.no_trade, "date"].unique())
    return price.index.difference(closed), audit


def read_sources(source, start, end):
    """Load warm-up and label observations before selecting signal dates."""
    source = Path(source)
    lower = pd.Timestamp(start) - pd.Timedelta(days=400)
    upper = pd.Timestamp(end) + pd.Timedelta(days=14)
    market = (
        pd.read_parquet(
            source / "market_data.parquet",
            filters=[("date", ">=", lower), ("date", "<=", upper)],
        )
        .rename_axis(index=["date", "company"])
        .reset_index()
    )
    jkp = (
        pd.read_parquet(
            source / "df_jkp.parquet",
            columns=[
                "eom",
                "iid",
                "id",
                "excntry",
                "obs_main",
                "exch_main",
                "common",
                "primary_sec",
                *JKP_FEATURES,
            ],
        )
        .rename_axis(index=["source_date", "company"])
        .reset_index()
    )
    return market, jkp.rename(columns={"id": "jkp_id", "excntry": "country"})


def monthly_features(jkp):
    """Ambiguous company/country/month matches remain explicit missing records."""
    jkp = jkp.copy()
    jkp["company"] = jkp.company.astype(str).str.zfill(6)
    jkp["eom"] = pd.to_datetime(jkp.eom).astype("datetime64[ns]")
    if jkp.eom.isna().any() or not jkp.eom.dt.is_month_end.all():
        raise ValueError("JKP eom must contain valid month-end dates")
    flags = ["obs_main", "exch_main", "common", "primary_sec"]
    jkp = jkp.loc[jkp[flags].eq(1).all(axis=1)].copy()
    keys = ["company", "country", "eom"]
    jkp["jkp_matches"] = jkp.groupby(keys, dropna=False).eom.transform("size")
    ambiguous = jkp.jkp_matches.ne(1)
    jkp.loc[ambiguous, [*JKP_FEATURES, "jkp_id"]] = np.nan
    jkp.loc[ambiguous, "iid"] = None
    # Duplicate groups now contain no selected issue or characteristic values.
    jkp = jkp.drop_duplicates(keys)
    jkp["available_date"] = jkp.eom + pd.Timedelta(days=1)
    return jkp[[*keys, "available_date", "jkp_matches", "jkp_id", "iid", *JKP_FEATURES]]


def prepare_daily(market, jkp, start, end, terminal_events=None):
    """Build a packed panel without selecting on future prices or labels.

    The schedule excludes country-wide inactivity dates. Missing quotes and
    finite returns are retained regardless of magnitude. Reviewed terminal events
    separately value holdings and cancel orders that can no longer execute.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError("start must not follow end")
    market = market.copy()
    market["date"] = pd.to_datetime(market.date).astype("datetime64[ns]")
    if market[["date", "company", "country"]].isna().any().any():
        raise ValueError("market dates, company IDs and countries must be present")
    market["company"] = market.company.astype(str).str.zfill(6)
    if market.duplicated(["date", "company"]).any():
        raise ValueError("duplicate market company/date")
    market = market.sort_values(["date", "company"]).reset_index(drop=True)
    scheduled, calendar_audit = trading_calendar(market)
    events = pd.DataFrame() if terminal_events is None else terminal_events.copy()
    if not events.empty:
        events["company"] = events.company.astype(str).str.zfill(6)
        events["stop_date"] = pd.to_datetime(events.stop_date)
        events["anchor_date"] = pd.to_datetime(events.anchor_date)
        if events.company.duplicated().any():
            raise ValueError("one terminal event per company required")
        if not (np.isfinite(events.proceeds_eur) & events.proceeds_eur.ge(0)).all():
            raise ValueError("terminal proceeds must be finite and nonnegative")
        if not events.anchor_date.lt(events.stop_date).all():
            raise ValueError("terminal anchor must precede the stop date")
        quotes = market.set_index(["company", "date"])
        keys = pd.MultiIndex.from_frame(events[["company", "anchor_date"]])
        source = "source_price" if "source_price" in market else "price"
        anchors = quotes.reindex(keys)
        if not np.isclose(anchors[source], events.anchor_price_eur, rtol=1e-8, atol=1e-10).all():
            raise ValueError("terminal event does not match source anchor price")
        events["adjusted_proceeds"] = events.proceeds_eur.to_numpy() * (
            anchors.price.to_numpy() / anchors[source].to_numpy()
        )
        positions = scheduled.searchsorted(events.stop_date)
        if (positions >= len(scheduled)).any():
            raise ValueError("terminal event outside loaded calendar")
        if "event_date" in events and not np.array_equal(
            pd.to_datetime(events.event_date).to_numpy(), scheduled[positions].to_numpy()
        ):
            raise ValueError("terminal valuation date does not match scheduled rebalance")
        events["event_date"] = scheduled[positions].to_numpy()
        if not np.array_equal(events.anchor_date.to_numpy(), scheduled[positions - 1].to_numpy()):
            raise ValueError("terminal event needs a quote at the preceding scheduled close")
        stop = market.company.map(events.set_index("company").stop_date)
        market = market.loc[stop.isna() | market.date.lt(stop)].copy()
    market = market.loc[market.date.isin(scheduled)].reset_index(drop=True)
    price = market.pivot(index="date", columns="company", values="price").reindex(scheduled)
    price = price.where(np.isfinite(price) & price.gt(0))
    terminal = pd.DataFrame(False, index=price.index, columns=price.columns)
    for event in events.itertuples():
        price.at[event.event_date, event.company] = event.adjusted_proceeds
        terminal.at[event.event_date, event.company] = True
    raw_return = price.div(price.shift(1)) - 1
    extreme = raw_return.abs().gt(0.5)
    extreme &= ~terminal
    if "split_corrected" in market:
        corrected = market.pivot(index="date", columns="company", values="split_corrected")
        # Approved events retain their economic return, even when genuinely large.
        extreme &= ~corrected.reindex_like(price).fillna(False).astype(bool)
    # Magnitude is a review signal, not evidence of a bad observation.
    returns = raw_return.where(np.isfinite(raw_return))
    calendar = price.index
    if len(calendar) < 66:
        raise ValueError("need at least 66 scheduled dates including warm-up")

    row = calendar.get_indexer(market.date)
    col = price.columns.get_indexer(market.company)

    def take(frame):
        return frame.to_numpy()[row, col]

    market["return_1"] = take(returns)
    for window in (5, 21):
        valid = returns.rolling(window).count().eq(window)
        market[f"return_{window}"] = take((price / price.shift(window) - 1).where(valid))
    complete = returns.rolling(63).count().eq(63)
    market["momentum_63_21"] = take((price.shift(21) / price.shift(63) - 1).where(complete))
    for window in (21, 63):
        market[f"volatility_{window}"] = take(returns.rolling(window).std())
    market["max_return_21"] = take(returns.rolling(21).max())
    market["risk_variance"] = np.maximum(market.volatility_63**2, 1e-8)

    schedule = pd.Series(calendar, index=calendar)
    market["entry_date"] = market.date.map(schedule.shift(-1))
    market["label_end"] = market.date.map(schedule.shift(-2))
    market["target_raw"] = take(raw_return.shift(-2))
    market["target"] = take(returns.shift(-2))
    market["target_basis"] = np.where(take(terminal.shift(-2, fill_value=False)),
                                     "terminal_distribution", "security_return")
    for event in events.itertuples():
        # An order reaching a stopped security is not filled; its allocation stays cash.
        cancelled = market.company.eq(event.company) & market.entry_date.ge(event.event_date)
        market.loc[cancelled, ["target", "target_raw"]] = 0.0
        market.loc[cancelled, "target_basis"] = "cancelled_order_cash"
    market["return_raw"] = take(raw_return)
    market["return_review"] = take(extreme)
    review = market.loc[market.return_review, ["date", "company", "price", "return_raw"]].copy()
    review["previous_price"] = take(price.shift(1))[market.return_review.to_numpy()]

    monthly = monthly_features(jkp)
    market = pd.merge_asof(
        market.sort_values("date"),
        monthly.sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        by=["company", "country"],
        direction="backward",
        tolerance=pd.Timedelta(days=62),
    )
    # Membership uses current/past market information only; JKP can be missing.
    eligible = np.isfinite(market[list(MARKET_FEATURES)]).all(axis=1)
    eligible &= market.date.between(start, end)
    selected = market.loc[eligible].copy()
    selected = selected.loc[selected.groupby("date").company.transform("size").ge(10)]
    selected = selected.sort_values(["date", "company"]).reset_index(drop=True)
    if selected.empty:
        raise ValueError("no dates with ten assets and sufficient daily history")

    signals = (*MARKET_FEATURES, *OPTIONAL_FEATURES)
    feature_names = []
    for feature in signals:
        values = selected[feature].where(np.isfinite(selected[feature]))
        ranks = values.groupby(selected.date).rank(method="average")
        counts = values.groupby(selected.date).transform("count")
        score = ndtri((ranks - 0.5) / counts)
        selected[feature + "_normal"] = score.fillna(0.0)
        feature_names.append(feature + "_normal")
    for feature in OPTIONAL_FEATURES:
        selected[feature + "_missing"] = ~np.isfinite(selected[feature])
        feature_names.append(feature + "_missing")

    ids, assets = pd.factorize(selected.company, sort=True)
    counts = selected.groupby("date", sort=True).size()
    panel = DynamicPanel(
        dates=counts.index.to_numpy(dtype="datetime64[D]"),
        offsets=np.r_[0, counts.to_numpy().cumsum()],
        assets=assets.to_numpy(dtype=str),
        asset_index=ids.astype(np.int32),
        features=selected[feature_names].to_numpy(dtype=np.float32, copy=True),
        feature_names=np.array(feature_names),
        targets=selected.target.to_numpy(dtype=np.float64, copy=True),
        entry_dates=selected.entry_date.to_numpy(dtype="datetime64[D]"),
        label_end=selected.label_end.to_numpy(dtype="datetime64[D]"),
        risk_variance=selected.risk_variance.to_numpy(dtype=np.float64, copy=True),
    )

    # An eligibility loss can be sold only at an observed next scheduled close.
    execution_prices = market.pivot(
        index="date", columns="company",
        values="source_price" if "source_price" in market else "price",
    )
    exits = []
    terminal_by_company = {e.company: e for e in events.itertuples()}
    for t in range(1, len(panel.dates)):
        old = set(panel.assets[panel.asset_index[panel.day(t - 1)]])
        current = set(panel.assets[panel.asset_index[panel.day(t)]])
        decision = pd.Timestamp(panel.dates[t])
        execution = pd.Timestamp(panel.entry_dates[panel.day(t)][0])
        for company in sorted(old - current):
            quote = execution_prices.at[execution, company] if execution in execution_prices.index else np.nan
            event = terminal_by_company.get(company)
            settled = event is not None and event.event_date <= execution
            exits.append(
                dict(
                    asset=company,
                    decision_date=decision,
                    execution_date=execution,
                    reason="corporate_action" if settled else "daily_ineligible",
                    exit_price=event.proceeds_eur if settled else quote,
                    event_date=event.event_date if settled else pd.NaT,
                    status=(
                        "terminal_settled" if settled else
                        "price_available" if np.isfinite(quote) and quote > 0 else "unresolved"
                    ),
                )
            )
    exits = pd.DataFrame(
        exits,
        columns=[
            "asset",
            "decision_date",
            "execution_date",
            "reason",
            "exit_price",
            "event_date",
            "status",
        ],
    )
    audit = selected[
        [
            "date",
            "company",
            "country",
            "eom",
            "available_date",
            "jkp_id",
            "iid",
            "jkp_matches",
            "entry_date",
            "label_end",
            "target_raw",
            "target",
            "target_basis",
        ]
    ].copy()
    audit["missing_features"] = selected[[f + "_missing" for f in OPTIONAL_FEATURES]].sum(axis=1)
    summary = dict(
        frequency="daily",
        return_basis="supplied-price returns; currency/adjustments unverified",
        calendar="skip dates with >=95% country inactivity (>=5 prior quotes); next scheduled close",
        no_trade_dates=int(
            calendar_audit.loc[
                calendar_audit.no_trade & calendar_audit.date.between(start, end), "date"
            ].nunique()
        ),
        universe="supplied companies with 63 consecutive usable returns; no additional size screen",
        jkp_join="company/country; eom+1 calendar day; backward as-of; maximum age 62 days",
        ambiguous_jkp="all matching characteristics missing; no arbitrary issue selection",
        features="date-wise inverse-normal ranks; optional missing scores zero with flags",
        feature_names=feature_names,
        outliers="finite returns retained; absolute returns above 50% flagged for review only",
        rows=len(panel.targets),
        dates=len(panel.dates),
        assets=len(panel.assets),
        missing_targets=int(np.isnan(panel.targets).sum()),
        ambiguous_jkp_rows=int(selected.jkp_matches.gt(1).sum()),
        unmatched_jkp_rows=int(selected.available_date.isna().sum()),
        extreme_return_rows=len(review),
        unresolved_exits=int(exits.status.eq("unresolved").sum()),
        terminal_events=len(events),
        zero_recovery_events=int(events.proceeds_eur.eq(0).sum()) if not events.empty else 0,
        terminal_target_rows=int(selected.target_basis.eq("terminal_distribution").sum()),
        cancelled_order_cash_rows=int(selected.target_basis.eq("cancelled_order_cash").sum()),
    )
    try:
        panel.validate_portfolio_dates(range(len(panel.dates)))
        summary["portfolio_dates_valid"] = True
    except ValueError as error:
        summary["portfolio_dates_valid"] = False
        summary["first_portfolio_error"] = str(error)
    return panel, audit, review, exits, summary
