import json

import numpy as np
import pandas as pd
import pytest
from btlight.data.panel import (
    ACCOUNTING_INPUTS,
    MARKET_INPUTS,
    DynamicPanel,
    Panel,
    load_data,
)


def write_source(path):
    (path / "daily").mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "currency": "EUR",
                "size_screen": {"fraction": 0.2},
            }
        )
    )
    rows = []
    for t, date in enumerate(pd.bdate_range("2020-01-01", periods=5)):
        for company in range(12):
            # One exit/re-entry and one new entrant; future survival is irrelevant.
            if (company == 0 and t == 2) or (company == 11 and t == 0):
                continue
            row = dict(
                gvkey=str(company),
                iid="01",
                signal_date=date,
                entry_date=date + pd.Timedelta(days=1),
                label_end=date + pd.Timedelta(days=2),
                target_return_eur=0.01,
                volatility_63=0.2,
            )
            row.update({f + "_normal": 0.4 for f in MARKET_INPUTS + ACCOUNTING_INPUTS})
            row.update({f + "_missing": False for f in ACCOUNTING_INPUTS})
            rows.append(row)
    data = pd.DataFrame(rows)
    data.loc[data.gvkey.eq("1"), "cashflow_assets_normal"] = np.nan
    data.loc[data.gvkey.eq("1"), "cashflow_assets_missing"] = True
    # A donor-imputed score remains a score, with original missingness preserved.
    data.loc[data.gvkey.eq("2"), "cash_assets_normal"] = 0.3
    data.loc[data.gvkey.eq("2"), "cash_assets_missing"] = True
    data.loc[data.gvkey.eq("3"), "target_return_eur"] = np.nan
    data.to_parquet(path / "daily/2020.parquet", index=False)
    return data


def test_dynamic_loader_entry_exit_missingness_and_roundtrip(tmp_path):
    source = tmp_path / "source"
    raw = write_source(source)
    panel = load_data(source)
    assert isinstance(panel, DynamicPanel)
    assert panel.features.shape == (len(raw), 24)
    asset = np.flatnonzero(panel.assets == "0:01")[0]
    assert asset in panel.asset_index[panel.day(0)]
    assert asset not in panel.asset_index[panel.day(2)]
    assert asset in panel.asset_index[panel.day(3)]
    cashflow = list(panel.feature_names).index("cashflow_assets_normal")
    missing = list(panel.feature_names).index("cashflow_assets_missing")
    unavailable = panel.asset_index == np.flatnonzero(panel.assets == "1:01")[0]
    assert (panel.features[unavailable, cashflow] == 0).all()
    assert (panel.features[unavailable, missing] == 1).all()
    assert (~panel.label_mask(0)).sum() == 1
    panel.save(tmp_path / "panel.npz")
    loaded = Panel.load(tmp_path / "panel.npz")
    np.testing.assert_array_equal(panel.features, loaded.features)
    np.testing.assert_array_equal(panel.targets, loaded.targets)
    # Future outcomes and future rows cannot change today's membership or features.
    raw.loc[raw.signal_date > raw.signal_date.min(), "return_1_normal"] = -100
    raw.target_return_eur = np.nan
    raw.to_parquet(source / "daily/2020.parquet", index=False)
    changed = DynamicPanel.from_cleaned(source)
    np.testing.assert_array_equal(panel.asset_index, changed.asset_index)
    np.testing.assert_array_equal(panel.features[panel.day(0)], changed.features[changed.day(0)])


def test_market_history_is_required_and_labels_are_purged_per_security(tmp_path):
    source = tmp_path / "source"
    raw = write_source(source)
    raw.loc[raw.gvkey.eq("4"), "momentum_63_21_normal"] = np.nan
    raw.to_parquet(source / "daily/2020.parquet", index=False)
    panel = DynamicPanel.from_cleaned(source)
    assert "4:01" not in panel.assets
    sl = panel.day(0)
    panel.label_end[sl.start] = np.datetime64("2020-01-10")
    mask = panel.label_mask(0, "2020-01-04")
    assert not mask[0] and mask[1:].any()
    assert not panel.label_mask(0, "2020-01-03").any()


def synthetic_dynamic():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2019-01-01", "2021-03-31").to_numpy(dtype="datetime64[D]")
    ids = [np.arange(12) if t % 3 else np.arange(1, 12) for t in range(len(dates))]
    counts = np.array([len(x) for x in ids])
    row_dates = np.repeat(dates, counts)
    targets = rng.normal(0, 0.01, counts.sum())
    targets[::13] = np.nan
    return DynamicPanel(
        dates=dates,
        offsets=np.r_[0, counts.cumsum()],
        assets=np.array([f"A{i}" for i in range(12)]),
        asset_index=np.concatenate(ids),
        features=rng.normal(size=(len(targets), 3)),
        feature_names=np.array(["a", "b", "c"]),
        targets=targets,
        entry_dates=row_dates + np.timedelta64(1, "D"),
        label_end=row_dates + np.timedelta64(2, "D"),
        risk_variance=np.full(len(targets), 0.0001),
    )


def test_dynamic_missing_outcomes_are_rejected():
    pytest.importorskip("torch")
    pytest.importorskip("cvxpylayers")
    from btlight.ml.training.e2e import run

    with pytest.raises(ValueError, match="Missing portfolio outcomes"):
        run(synthetic_dynamic(), epochs=1, first_test="2021-01-01", smoke=True)


def test_marginal_moments_match_joint_gaussian():
    torch = pytest.importorskip("torch")
    from btlight.ml.model.return_predictor import GaussianReturnPredictor

    model = GaussianReturnPredictor(3).double().eval()
    x = torch.randn(12, 3, dtype=torch.float64)
    mean, variance = model.predict_moments(x)
    joint = model.predict_distribution(x)
    torch.testing.assert_close(mean, joint.mean)
    torch.testing.assert_close(variance, joint.variance)


def test_holdings_alignment_sales_and_gradients():
    torch = pytest.importorskip("torch")
    from btlight.ml.training.portfolio_data import align_holdings

    old = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="Unresolved holdings"):
        align_holdings([1, 2, 3], old, [3, 2, 4])
    aligned, sold = align_holdings([1, 2, 3], old, [3, 2, 4], {1})
    torch.testing.assert_close(aligned, torch.tensor([0.5, 0.3, 0.0], dtype=torch.float64))
    new = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float64)
    turnover = ((new - aligned).abs().sum() + sold) / 2
    assert turnover.item() == pytest.approx(0.3)
    turnover.backward()
    assert old.grad[0].item() == 0.5
    with pytest.raises(ValueError, match="cannot also be allocated"):
        align_holdings([1, 2, 3], old, [1, 2, 3], {1})


def test_diagonal_layer_matches_dense_and_differentiates():
    torch = pytest.importorskip("torch")
    from btlight.optimization.differentiable import PortfolioLayer

    mu = torch.tensor([0.03, 0.02, 0.04], dtype=torch.float64, requires_grad=True)
    sd = torch.tensor([0.3, 0.4, 0.35], dtype=torch.float64, requires_grad=True)
    previous = torch.ones(3, dtype=torch.float64) / 3
    diagonal = PortfolioLayer(3, cap=0.8, diagonal=True)
    dense = PortfolioLayer(3, cap=0.8)
    torch.testing.assert_close(
        diagonal(mu, sd, previous),
        dense(mu, torch.diag(sd), previous),
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.autograd.gradcheck(
        lambda m, s: diagonal(m, s, previous), (mu, sd), eps=1e-3, atol=2e-3, rtol=2e-2
    )

    # Independent interior optimum verifies the variance derivative analytically.
    def exact(s):
        variance = s.square()
        multiplier = ((mu / variance).sum() - 2) / (1 / variance).sum()
        return (mu - multiplier) / (2 * variance)

    torch.testing.assert_close(
        torch.autograd.functional.jacobian(lambda s: diagonal(mu, s, previous), sd),
        torch.autograd.functional.jacobian(exact, sd),
        atol=1e-4,
        rtol=1e-3,
    )


def test_exit_lookup_requires_matching_execution(tmp_path):
    from btlight.ml.training.portfolio_data import exit_lookup

    p = synthetic_dynamic()
    p.assets = np.array([f"{i}:01" for i in range(12)])
    date = pd.Timestamp(p.dates[3])
    entry = pd.Timestamp(p.entry_dates[p.day(3)][0])
    audit = pd.DataFrame(
        {
            "gvkey": ["0", "1"],
            "iid": ["01", "01"],
            "decision_date": [date, date],
            "execution_date": [entry, entry + pd.Timedelta(days=1)],
            "reason": ["bottom_20pct"] * 2,
            "status": ["price_available"] * 2,
            "exit_price_eur": [10.0, 20.0],
        }
    )
    audit.to_parquet(tmp_path / "exits.parquet")
    assert exit_lookup(p, tmp_path / "exits.parquet")[3] == {0}


@pytest.mark.parametrize(
    "prediction,strategy",
    [("point", "e2e"), ("gaussian", "e2e"), ("gaussian", "pto"), ("point", "equal_weight")],
)
def test_dynamic_e2e_with_confirmed_exits(tmp_path, prediction, strategy):
    from btlight.ml.training.e2e import run

    p = synthetic_dynamic()
    p.targets = np.nan_to_num(p.targets, nan=0.001)  # Synthetic known outcomes only.
    calendar = pd.bdate_range(p.dates[0], periods=len(p.dates) + 2).to_numpy(dtype="datetime64[D]")
    counts = np.diff(p.offsets)
    p.entry_dates = np.repeat(calendar[1:-1], counts)
    p.label_end = np.repeat(calendar[2:], counts)
    exits = {
        t: set(p.asset_index[p.day(t - 1)]) - set(p.asset_index[p.day(t)])
        for t in range(1, len(p.dates))
    }
    result = run(
        p,
        epochs=1,
        smoke=True,
        first_test="2021-01-01",
        risk="variance",
        prediction="equal_weight" if strategy == "equal_weight" else prediction,
        finetune=strategy != "pto",
        cost=0.001,
        executed_exits=exits,
        distribution_output=(tmp_path / "dynamic.npz" if prediction == "gaussian" else None),
    )
    assert result.observations.iloc[0] == 8
    assert np.isfinite(result.annualized_return.iloc[0])
    if strategy == "equal_weight":
        from btlight.ml.training.portfolio_data import experiment_splits

        _, _, _, test = next(experiment_splits(p, "2021-01-01", range(len(p.dates)), True))
        previous = {}
        returns = []
        for t in test:
            sl = p.day(t)
            ids = p.asset_index[sl]
            current = dict.fromkeys(ids, 1 / len(ids))
            turnover = sum(abs(current.get(i, 0) - previous.get(i, 0))
                           for i in current.keys() | previous.keys()) / 2
            returns.append(p.targets[sl].mean() - 0.001 * turnover)
            previous = current
        assert result.annualized_return.iloc[0] == pytest.approx(np.mean(returns) * 252)
    if prediction == "gaussian":
        with np.load(tmp_path / "dynamic.npz", allow_pickle=False) as saved:
            assert len(np.unique(np.diff(saved["offsets"]))) > 1
            for start, end in zip(saved["offsets"][:-1], saved["offsets"][1:]):
                assert saved["weights"][start:end].sum() == pytest.approx(1.0, abs=1e-6)
