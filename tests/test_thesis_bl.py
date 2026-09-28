import pytest

pytest.importorskip("torch")
pytest.importorskip("cvxpylayers")

import numpy as np
import pandas as pd
import torch
from btlight.estimation.differentiable_bl import BLUpdate, historical_bl_inputs, load_market_weights
from btlight.data.panel import synthetic_panel
from btlight.optimization.differentiable import PortfolioLayer
from btlight.ml.training.wang_e2e import run


def test_bl_formula_limits_and_composed_gradients():
    covariance = torch.tensor([[0.04, 0.01], [0.01, 0.09]], dtype=torch.float64)
    prior = torch.tensor([0.02, 0.03], dtype=torch.float64)
    views = torch.tensor([0.03, 0.02], dtype=torch.float64, requires_grad=True)
    variance = torch.tensor([0.03, 0.02], dtype=torch.float64, requires_grad=True)
    bl = BLUpdate(variance, tau=0.5)
    precision = torch.linalg.inv(0.5 * covariance)
    omega_precision = torch.diag(1 / variance)
    expected = torch.linalg.solve(
        precision + omega_precision, precision @ prior + omega_precision @ views
    )
    torch.testing.assert_close(bl(views, prior, covariance), expected)
    # The existing NumPy implementation provides an independent migration check.
    from btlight.estimation.black_litterman import bl_posterior_mu_sigma

    reference, _ = bl_posterior_mu_sigma(
        pd.Series(prior.numpy()),
        pd.DataFrame(covariance.numpy()),
        np.eye(2),
        views.detach().numpy(),
        0.5 * covariance.numpy(),
        np.diag(variance.detach().numpy()),
    )
    np.testing.assert_allclose(expected.detach().numpy(), reference.to_numpy())
    torch.testing.assert_close(bl(views, prior, covariance, torch.ones(2) * 1e8), prior)
    torch.testing.assert_close(bl(views, prior, covariance, torch.ones(2) * 1e-12), views)
    with pytest.raises(ValueError, match="positive"):
        BLUpdate(torch.zeros(2))
    assert list(bl.parameters()) == []

    qp = PortfolioLayer(2, risk_aversion=1, cap=0.9, risk="variance")
    factor = torch.linalg.cholesky(covariance).T
    previous = torch.ones(2, dtype=torch.float64) / 2

    def composed(mu, omega):
        posterior = bl(mu, prior, covariance, omega)
        return qp(posterior, factor, previous)

    assert torch.autograd.gradcheck(composed, (views, variance), eps=1e-5, atol=2e-3, rtol=2e-2)
    gradients = torch.autograd.grad(composed(views, variance)[0], (views, variance))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_prior_alignment_and_no_future_information(tmp_path):
    panel = synthetic_panel(periods=50, assets=3)
    market_raw = (panel.raw_returns - panel.targets)[:, 0]
    factors = pd.DataFrame(
        {"Mkt-RF": market_raw - 0.0001, "RF": 0.0001}, index=pd.to_datetime(panel.label_end)
    )
    weights = pd.DataFrame(
        [[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]],
        index=pd.to_datetime(panel.dates[[0, 30]]),
        columns=panel.assets,
    )
    weights.index.name = "date"
    path = tmp_path / "weights.csv"
    weights[weights.columns[::-1]].to_csv(path)
    loaded = load_market_weights(path, panel.assets)
    priors, covariances = historical_bl_inputs(panel, loaded, factors, lookback=10)
    t = 20
    ids = np.flatnonzero(panel.label_end <= panel.dates[t])[-10:]
    expected_cov = np.cov(panel.raw_returns[ids] - 0.0001, rowvar=False, ddof=1)
    expected = 2.5 * expected_cov @ weights.iloc[0].to_numpy() - factors["Mkt-RF"].iloc[ids].mean()
    np.testing.assert_allclose(priors[t], expected)
    np.testing.assert_allclose(covariances[t], expected_cov)
    panel.raw_returns[19:] += 0.5
    panel.targets[19:] += 0.5
    changed, _ = historical_bl_inputs(panel, loaded, factors, lookback=10)
    torch.testing.assert_close(priors[t], changed[t])
    assert 0 not in priors


@pytest.mark.parametrize("mode", ["constant", "historical"])
def test_fixed_bl_training_runs(mode):
    result = run(
        synthetic_panel(), epochs=1, smoke=True, risk="variance", bl_mode=mode, error_window=20
    )
    assert result.observations.iloc[0] == 8
    assert result.bl_mode.iloc[0] == mode
    assert np.isfinite(result.select_dtypes(include="number").to_numpy()).all()
