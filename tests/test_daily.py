import numpy as np
import pandas as pd
import pytest

from btlight.data.daily import JKP_FEATURES, prepare_daily, read_sources
from btlight.data.panel import Panel


def test_systematic_calendar_skips_country_closures():
    from btlight.data.daily import trading_calendar

    market, jkp, dates = source()
    market.loc[market.date.eq(dates[100]), "liquidity"] = 0
    market = market.loc[~(market.company.eq("000000") & market.date.eq(dates[110]))]
    schedule, audit = trading_calendar(market)
    assert dates[100] not in schedule
    assert dates[110] in schedule
    assert audit.loc[audit.date.eq(dates[100]), "no_trade"].all()
    panel, rows, *_ = prepare_daily(market, jkp, dates[64], dates[-3])
    assert dates[100] not in pd.DatetimeIndex(panel.dates)
    before = rows[rows.date.eq(dates[99]) & rows.company.eq("000001")].iloc[0]
    assert before.entry_date == dates[101]
    assert before.label_end == dates[102]
    # The earlier allocation is held across the skipped day; its return is retained.
    held = rows[rows.date.eq(dates[98]) & rows.company.eq("000001")].iloc[0]
    assert held.entry_date == dates[99] and held.label_end == dates[101]
    prices = market[market.company.eq("000001")].set_index("date").price
    assert held.target == pytest.approx(prices[dates[101]] / prices[dates[99]] - 1)


def source(periods=150):
    dates = pd.bdate_range("2019-01-01", periods=periods)
    rng = np.random.default_rng(7)
    companies = [f"{i:06}" for i in range(12)]
    prices = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, (periods, 12)), axis=0)
    market = pd.DataFrame(
        {
            "date": np.repeat(dates, 12),
            "company": companies * periods,
            "price": prices.ravel(),
            "market_cap": 1e9,
            "liquidity": 1e6,
            "country": "FRA",
        }
    )
    rows = []
    for eom in pd.date_range("2018-12-31", dates[-1], freq="ME"):
        for i, company in enumerate(companies):
            row = dict(
                company=company,
                eom=eom,
                source_date=eom - pd.Timedelta(days=5),
                country="FRA",
                iid="01",
                jkp_id=i,
                obs_main=1,
                exch_main=1,
                common=1,
                primary_sec=1,
            )
            row.update({f: i + eom.month for f in JKP_FEATURES})
            rows.append(row)
    return market, pd.DataFrame(rows), dates


def test_monthly_availability_ambiguity_and_roundtrip(tmp_path):
    market, jkp, dates = source()
    # Foreign listing must not join; same-country duplicates are left missing.
    row = jkp[jkp.company.eq("000000") & jkp.eom.eq("2019-04-30")].copy()
    foreign = row.assign(country="USA", jkp_id=100, be_me=9999)
    duplicate = row.assign(company="000001", jkp_id=200, be_me=8888)
    jkp = pd.concat([jkp, foreign, duplicate], ignore_index=True)
    panel, audit, _, _, _ = prepare_daily(market, jkp, "2019-04-01", dates[-3])
    before = audit[audit.date.eq("2019-04-30") & audit.company.eq("000000")].iloc[0]
    after = audit[audit.date.eq("2019-05-01") & audit.company.eq("000000")].iloc[0]
    assert before.eom == pd.Timestamp("2019-03-31")
    assert after.eom == pd.Timestamp("2019-04-30") and after.jkp_matches == 1
    ambiguous = audit[audit.date.eq("2019-05-01") & audit.company.eq("000001")].iloc[0]
    assert ambiguous.jkp_matches == 2 and pd.isna(ambiguous.jkp_id)
    i = ambiguous.name
    assert panel.features[i, list(panel.feature_names).index("be_me_missing")] == 1
    assert panel.features[i, list(panel.feature_names).index("be_me_normal")] == 0
    assert (audit.available_date <= audit.date).all()
    panel.validate_portfolio_dates(range(len(panel.dates)))
    panel.save(tmp_path / "daily.npz")
    loaded = Panel.load(tmp_path / "daily.npz")
    np.testing.assert_array_equal(loaded.features, panel.features)


def test_missing_future_quotes_do_not_select_assets_or_fill_returns(tmp_path):
    from btlight.ml.training.portfolio_data import exit_lookup

    market, jkp, dates = source()
    base, audit, *_ = prepare_daily(market, jkp, dates[64], dates[-3])
    changed = market.loc[~(market.company.eq("000000") & market.date.eq(dates[100]))]
    panel, changed_audit, _, exits, _ = prepare_daily(changed, jkp, dates[64], dates[-3])
    original = audit[audit.date <= dates[98]]
    earlier = changed_audit[changed_audit.date <= dates[98]]
    pd.testing.assert_frame_equal(original[["date", "company"]], earlier[["date", "company"]])
    np.testing.assert_array_equal(base.features[: len(original)], panel.features[: len(earlier)])
    unknown = changed_audit.date.eq(dates[98]) & changed_audit.company.eq("000000")
    assert changed_audit.loc[unknown, "target"].isna().all()
    assert len(exits) > 0
    exits.to_parquet(tmp_path / "exits.parquet", index=False)
    confirmed = exit_lookup(panel, tmp_path / "exits.parquet")
    exit_t = int(np.flatnonzero(panel.dates == dates[100].to_datetime64())[0])
    asset = int(np.flatnonzero(panel.assets == "000000")[0])
    assert confirmed[exit_t] == {asset}
    exits["execution_date"] += pd.Timedelta(days=1)
    exits.to_parquet(tmp_path / "delayed.parquet", index=False)
    assert not exit_lookup(panel, tmp_path / "delayed.parquet")[exit_t]
    with pytest.raises(ValueError, match="Missing portfolio outcomes"):
        panel.validate_portfolio_dates(range(len(panel.dates)))


def test_return_horizon_outliers_and_stale_jkp():
    market, jkp, dates = source()
    jkp = jkp[jkp.eom <= "2019-03-31"]
    market.loc[market.company.eq("000000") & market.date.eq(dates[100]), "price"] *= 100
    panel, audit, review, _, _ = prepare_daily(market, jkp, dates[64], dates[-3])
    assert len(review) == 2 and review.return_raw.abs().gt(0.5).all()
    affected = audit.company.eq("000000") & audit.label_end.eq(dates[100])
    assert audit.loc[affected, "target"].gt(50).all()
    assert np.isfinite(panel.targets).all()
    i = audit.index[audit.company.eq("000001") & audit.date.eq(dates[80])][0]
    prices = market[market.company.eq("000001")].set_index("date").price
    assert panel.targets[i] == pytest.approx(prices[dates[82]] / prices[dates[81]] - 1)
    assert pd.Timestamp(panel.entry_dates[i]) == dates[81]
    stale = audit[audit.date > "2019-06-02"]
    assert stale.available_date.isna().all()


def test_parquet_index_collision_and_training(tmp_path):
    from btlight.ml.training.e2e import run
    from btlight.ml.training.portfolio_data import exit_lookup

    market, jkp, dates = source(600)
    market.rename(columns={"company": "id"}).set_index(["date", "id"]).to_parquet(
        tmp_path / "market_data.parquet"
    )
    jkp = jkp.rename(columns={"jkp_id": "id", "country": "excntry"})
    jkp = jkp.set_index(["source_date", "company"]).rename_axis(index=["date", "id"])
    jkp.to_parquet(tmp_path / "df_jkp.parquet")
    market, jkp = read_sources(tmp_path, "2019-04-01", dates[-3])
    panel, _, _, exits, _ = prepare_daily(market, jkp, "2019-04-01", dates[-3])
    exits.to_parquet(tmp_path / "exits.parquet", index=False)
    assert exit_lookup(panel, tmp_path / "exits.parquet") == {}
    result = run(
        panel,
        prediction="gaussian",
        pretrain=False,
        epochs=1,
        smoke=True,
        first_test="2021-01-01",
    )
    assert np.isfinite(result.test_nll_per_asset.iloc[0])


def test_split_checker_matches_issues_and_both_split_directions():
    from btlight.data.daily import check_stock_splits

    dates = pd.bdate_range('2020-01-01', periods=3)
    market = pd.DataFrame(dict(company=['000001'] * 3, date=dates, price=[10., 10000., 5000.]))
    security = pd.DataFrame(dict(gvkey=['000001'] * 3, iid=['01'] * 3,
                                 datadate=dates, prccd=market.price, ajexdi=[.001, 1., .5]))
    original = market.copy(deep=True)
    result = check_stock_splits(market, security)
    assert result.status.tolist() == ['matched_candidate'] * 2
    np.testing.assert_allclose(result.proposed_return, 0)
    np.testing.assert_allclose(result.factor_ratio, [1000, .5])
    pd.testing.assert_frame_equal(market, original)
    # A matching second issue prevents arbitrary selection.
    ambiguous = check_stock_splits(market, pd.concat([security, security.assign(iid='02')]))
    assert ambiguous.proposed_return.isna().all()
    # Already adjusted or differently denominated supplied prices must not be corrected.
    mismatch = check_stock_splits(market.assign(price=10.), security)
    assert mismatch.proposed_return.isna().all()
    unchanged = check_stock_splits(market, security.assign(ajexdi=1.))
    assert unchanged.empty


def test_return_signals_are_rank_normalized_and_targets_remain_returns():
    from scipy.special import ndtri

    market, jkp, dates = source()
    panel, audit, *_ = prepare_daily(market, jkp, dates[64], dates[-3])
    sl = panel.day(0)
    scores = panel.features[sl, list(panel.feature_names).index('return_1_normal')]
    np.testing.assert_allclose(np.sort(scores), ndtri((np.arange(12) + .5) / 12), rtol=1e-6)
    np.testing.assert_allclose(panel.targets, audit.target_raw)


def test_split_validation_and_application():
    from btlight.data.daily import check_stock_splits, validate_stock_splits, apply_stock_splits

    dates = pd.bdate_range('2020-01-01', periods=4)
    market = pd.DataFrame(dict(company=['000001'] * 4, date=dates, price=[10., 10000., 5000., 5500.]))
    security = pd.DataFrame(dict(gvkey=['000001'] * 4, iid=['01'] * 4,
        datadate=dates, prccd=market.price, ajexdi=[.001, 1., .5, .5],
        cshoc=[1000000., 1000., 2000., 2000.], curcdd='EUR'))
    decisions = validate_stock_splits(check_stock_splits(market, security), security)
    assert decisions.decision.eq('apply').all()
    adjusted = apply_stock_splits(market, decisions)
    np.testing.assert_allclose(adjusted.price, [10, 10, 10, 11])
    np.testing.assert_array_equal(adjusted.source_price, market.price)
    np.testing.assert_array_equal(market.price, [10, 10000, 5000, 5500])
    with pytest.raises(ValueError, match='already'):
        apply_stock_splits(adjusted, decisions)
    with pytest.raises(ValueError, match='endpoint'):
        apply_stock_splits(market.assign(price=market.price * 2), decisions)
    bad = security.assign(cshoc=1000000.)
    assert validate_stock_splits(check_stock_splits(market, bad), bad).decision.eq('not_applied').all()
    bad = security.copy()
    bad.loc[1, 'curcdd'] = 'USD'
    assert validate_stock_splits(check_stock_splits(market, bad), bad).decision.eq('not_applied').all()


def test_adjusted_history_rebuilds_targets_features_and_risk():
    from btlight.data.daily import apply_stock_splits

    market, jkp, dates = source()
    base, *_ = prepare_daily(market, jkp, dates[64], dates[-3])
    split = market.company.eq('000000') & market.date.ge(dates[100])
    market.loc[split, 'price'] *= 1000
    series = market[market.company.eq('000000')].set_index('date').price
    decisions = pd.DataFrame([dict(company='000000', date=dates[100], previous_date=dates[99],
        price=series[dates[100]], previous_price=series[dates[99]], factor_ratio=1000., decision='apply')])
    adjusted = apply_stock_splits(market, decisions)
    panel, _, review, _, _ = prepare_daily(adjusted, jkp, dates[64], dates[-3])
    assert review.empty
    np.testing.assert_array_equal(panel.asset_index, base.asset_index)
    np.testing.assert_allclose(panel.features, base.features, atol=1e-7)
    np.testing.assert_allclose(panel.targets, base.targets, atol=1e-12)
    np.testing.assert_allclose(panel.risk_variance, base.risk_variance, atol=1e-12)


def test_verified_adjustment_retains_large_economic_return():
    from btlight.data.daily import apply_stock_splits

    market, jkp, dates = source()
    affected = market.company.eq('000000') & market.date.ge(dates[100])
    market.loc[affected, 'price'] *= 1600  # 1000:1 split plus a real gain.
    prices = market[market.company.eq('000000')].set_index('date').price
    expected = prices[dates[100]] / prices[dates[99]] / 1000 - 1
    assert expected > .5
    events = pd.DataFrame([dict(company='000000', date=dates[100], previous_date=dates[99],
        price=prices[dates[100]], previous_price=prices[dates[99]], factor_ratio=1000., decision='apply')])
    panel, audit, review, *_ = prepare_daily(apply_stock_splits(market, events), jkp, dates[64], dates[-3])
    target = audit[audit.company.eq('000000') & audit.label_end.eq(dates[100])].target.iloc[0]
    assert target == pytest.approx(expected)
    assert review.empty
    assert np.isfinite(panel.features).all()


@pytest.mark.parametrize('recovery', [0.0, 1.08])
def test_terminal_proceeds_loss_and_cancelled_orders(tmp_path, recovery):
    from btlight.ml.training.portfolio_data import exit_lookup

    market, jkp, dates = source()
    original = market.copy(deep=True)
    anchor = market.loc[market.company.eq('000000') & market.date.eq(dates[99]), 'price'].iloc[0]
    events = pd.DataFrame([dict(company='000000', stop_date=dates[100],
                                anchor_date=dates[99], anchor_price_eur=anchor,
                                proceeds_eur=anchor * recovery)])
    panel, audit, review, exits, summary = prepare_daily(market, jkp, dates[64], dates[-3], events)
    stock = audit[audit.company.eq('000000')].set_index('date')
    assert stock.loc[dates[98], 'target'] == pytest.approx(recovery - 1)
    assert stock.loc[dates[98], 'target_basis'] == 'terminal_distribution'
    assert stock.loc[dates[99], 'target'] == 0
    assert stock.loc[dates[99], 'target_basis'] == 'cancelled_order_cash'
    assert stock.index.max() == dates[99]
    assert review.empty
    assert summary['missing_targets'] == 0
    panel.validate_portfolio_dates(range(len(panel.dates)))
    panel.save(tmp_path / 'terminal.npz')
    np.testing.assert_equal(Panel.load(tmp_path / 'terminal.npz').targets, panel.targets)
    exits.to_parquet(tmp_path / 'exits.parquet', index=False)
    lookup = exit_lookup(panel, tmp_path / 'exits.parquet')
    t = int(np.flatnonzero(panel.dates == dates[100].to_datetime64())[0])
    asset = int(np.flatnonzero(panel.assets == '000000')[0])
    assert lookup[t] == {asset}
    pd.testing.assert_frame_equal(market, original)
    # Future terminal outcomes cannot change earlier membership or input features.
    base, baseline, *_ = prepare_daily(market, jkp, dates[64], dates[-3])
    before = audit.date.lt(dates[100])
    n = before.sum()
    pd.testing.assert_frame_equal(audit.loc[before, ['date', 'company']], baseline.iloc[:n][['date', 'company']])
    np.testing.assert_allclose(panel.features[:n], base.features[:n])
    with pytest.raises(ValueError, match='anchor price'):
        prepare_daily(market, jkp, dates[64], dates[-3], events.assign(anchor_price_eur=1))
    with pytest.raises(ValueError, match='nonnegative'):
        prepare_daily(market, jkp, dates[64], dates[-3], events.assign(proceeds_eur=-1))


def test_terminal_proceeds_respect_prior_split_scale():
    from btlight.data.daily import apply_stock_splits

    market, jkp, dates = source()
    market.loc[market.company.eq('000000') & market.date.ge(dates[90]), 'price'] *= 1000
    prices = market[market.company.eq('000000')].set_index('date').price
    splits = pd.DataFrame([dict(company='000000', date=dates[90], previous_date=dates[89],
        price=prices[dates[90]], previous_price=prices[dates[89]], factor_ratio=1000., decision='apply')])
    terminal = pd.DataFrame([dict(company='000000', stop_date=dates[100], anchor_date=dates[99],
        anchor_price_eur=prices[dates[99]], proceeds_eur=prices[dates[99]] * 1.1)])
    _, audit, *_ = prepare_daily(apply_stock_splits(market, splits), jkp, dates[64], dates[-3], terminal)
    row = audit[audit.company.eq('000000') & audit.label_end.eq(dates[100])].iloc[0]
    assert row.target == pytest.approx(.1)


def test_e2e_smoke_accepts_terminal_loss(tmp_path):
    from btlight.ml.training.e2e import run
    from btlight.ml.training.portfolio_data import exit_lookup

    market, jkp, dates = source(600)
    stop = pd.Timestamp('2021-01-08')
    anchor_date = dates[dates.get_loc(stop) - 1]
    anchor = market.loc[market.company.eq('000000') & market.date.eq(anchor_date), 'price'].iloc[0]
    events = pd.DataFrame([dict(company='000000', stop_date=stop, anchor_date=anchor_date,
                                anchor_price_eur=anchor, proceeds_eur=0.)])
    panel, audit, _, exits, _ = prepare_daily(market, jkp, '2019-04-01', dates[-3], events)
    assert audit.target.eq(-1).sum() == 1
    exits.to_parquet(tmp_path / 'exits.parquet', index=False)
    result = run(panel, prediction='gaussian', pretrain=False,
                 epochs=1, smoke=True, first_test='2021-01-01',
                 executed_exits=exit_lookup(panel, tmp_path / 'exits.parquet'))
    assert np.isfinite(result.test_nll_per_asset.iloc[0])


def test_split_matching_converts_both_endpoint_currencies():
    from btlight.data.daily import check_stock_splits

    dates = pd.bdate_range('2020-01-01', periods=2)
    security = pd.DataFrame(dict(gvkey=['000001']*2, iid=['01']*2, datadate=dates,
                                 prccd=[100., 50.], ajexdi=[1., .5], curcdd='DKK'))
    market = pd.DataFrame(dict(company=['000001']*2, date=dates, price=[10., 5.5]))
    fx = pd.DataFrame(dict(fromcurd='GBP', tocurd=['EUR','DKK','EUR','DKK'],
                          datadate=[dates[0],dates[0],dates[1],dates[1]], exratd=[1.,10.,1.1,10.]))
    assert check_stock_splits(market, security).status.eq('unverified_issue').all()
    checked = check_stock_splits(market, security, fx)
    assert checked.status.eq('matched_candidate').all()
    assert checked.proposed_return.iloc[0] == pytest.approx(.1)
    missing = check_stock_splits(market, security, fx.iloc[2:])
    assert missing.status.eq('unverified_issue').all()


def test_distribution_value_is_included_once():
    from btlight.data.daily import apply_stock_splits

    market, jkp, dates = source()
    stock = market.company.eq('000000')
    market.loc[stock & market.date.ge(dates[100]), 'price'] *= .4
    p = market.loc[stock].set_index('date').price
    distribution = 60.
    events = pd.DataFrame([dict(company='000000', date=dates[100], previous_date=dates[99],
        price=p[dates[100]], previous_price=p[dates[99]],
        factor_ratio=p[dates[100]] / (p[dates[100]] + distribution), decision='apply')])
    _, audit, *_ = prepare_daily(apply_stock_splits(market, events), jkp, dates[64], dates[-3])
    a = audit[audit.company.eq('000000')].set_index('label_end')
    assert a.loc[dates[100], 'target'] == pytest.approx((p[dates[100]] + distribution) / p[dates[99]] - 1)
    assert a.loc[dates[101], 'target'] == pytest.approx(p[dates[101]] / p[dates[100]] - 1)
