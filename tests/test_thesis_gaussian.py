import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpylayers")

from btlight.data.panel import synthetic_panel
from btlight.ml.model.return_predictor import GaussianReturnPredictor, gaussian_nll
from btlight.ml.training.portfolio_data import correlation_factors
from btlight.ml.training.e2e import run
from btlight.optimization.differentiable import PortfolioLayer


def test_gaussian_density_covariance_and_variance_floor():
    model = GaussianReturnPredictor(2, depth=0, initial_variance=0.02).double()
    x = torch.eye(2, dtype=torch.float64)
    correlation = torch.tensor([[1.0, 0.4], [0.4, 1.0]], dtype=torch.float64)
    distribution = model.predict_distribution(x, torch.linalg.cholesky(correlation))
    torch.testing.assert_close(distribution.variance, torch.full((2,), 0.02, dtype=torch.float64))
    torch.testing.assert_close(distribution.covariance_matrix, 0.02 * correlation)
    y = torch.tensor([0.1, -0.2], dtype=torch.float64)
    residual = (y - distribution.mean).detach().numpy()
    covariance = distribution.covariance_matrix.detach().numpy()
    reference = (
        2 * np.log(2 * np.pi)
        + np.linalg.slogdet(covariance)[1]
        + residual @ np.linalg.solve(covariance, residual)
    ) / 4
    np.testing.assert_allclose(gaussian_nll(distribution, y).detach(), reference)
    with torch.no_grad():
        model.variance_head.bias.fill_(-1000)
    assert (model.predict_distribution(x).variance >= model.variance_floor * (1 - 1e-12)).all()
    with pytest.raises(ValueError, match="initial_variance"):
        GaussianReturnPredictor(2, initial_variance=1e-9)


def test_covariance_gradient_through_portfolio_matches_finite_differences():
    correlation = torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
    chol = torch.linalg.cholesky(correlation)
    mu = torch.tensor([0.02, 0.01], dtype=torch.float64, requires_grad=True)
    variance = torch.tensor([0.04, 0.06], dtype=torch.float64, requires_grad=True)
    previous = torch.full((2,), 0.5, dtype=torch.float64)
    layer = PortfolioLayer(2, cap=0.9, risk="variance")

    def allocate(mean, var):
        factor = (var.sqrt()[:, None] * chol).T
        return layer(mean, factor, previous)

    assert torch.autograd.gradcheck(allocate, (mu, variance), eps=1e-5, atol=2e-3, rtol=2e-2)
    assert ((allocate(mu, variance) > 0.05) & (allocate(mu, variance) < 0.85)).all()

    model = GaussianReturnPredictor(2, depth=0, initial_variance=0.04).double()
    distribution = model.predict_distribution(torch.eye(2, dtype=torch.float64), chol)
    weights = layer(distribution.mean, distribution.scale_tril.T, previous)
    realized = torch.tensor([0.1, -0.1], dtype=torch.float64)
    fixed_factor = 0.2 * torch.eye(2, dtype=torch.float64)
    # No likelihood term here: verify the allocation path reaches both heads.
    layer.loss(weights, realized, fixed_factor, previous).backward()
    for head in (model.network[-1], model.variance_head):
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 1e-8


def test_correlation_history_is_causal_and_handles_constant_assets():
    panel = synthetic_panel(periods=40, assets=3)
    panel.risk_returns[:, 0] = 0
    factors = correlation_factors(panel, lookback=10, shrinkage=0.2)
    correlation = factors[20] @ factors[20].T
    torch.testing.assert_close(correlation.diag(), torch.ones(3, dtype=torch.float64))
    assert torch.linalg.eigvalsh(correlation).min() >= 0.2 - 1e-12
    panel.risk_returns[21:] = 999
    panel.targets[:] = -999
    changed = correlation_factors(panel, lookback=10, shrinkage=0.2)
    torch.testing.assert_close(changed[20], factors[20])
    assert 8 not in factors and 9 in factors


def test_gaussian_nll_gradient_learns_larger_variance_for_larger_errors():
    model = GaussianReturnPredictor(2, depth=0, initial_variance=0.01).double()
    x = torch.eye(2, dtype=torch.float64)
    with torch.no_grad():
        model.network[-1].weight.zero_()
    distribution = model.predict_distribution(x)
    gaussian_nll(distribution, torch.tensor([0.01, 0.3], dtype=torch.float64)).backward()
    # Gradient descent reduces variance for the small error and increases it for the large one.
    assert model.variance_head.weight.grad[0, 0] > 0
    assert model.variance_head.weight.grad[0, 1] < 0


def test_gaussian_training_and_forecast_diagnostics(tmp_path):
    path = tmp_path / "distribution.npz"
    result = run(
        synthetic_panel(),
        epochs=1,
        smoke=True,
        prediction="gaussian",
        risk="variance",
        distribution_output=path,
    )
    assert result.observations.iloc[0] == 8
    assert np.isfinite(result.select_dtypes(include="number").to_numpy()).all()
    with np.load(path, allow_pickle=False) as saved:
        assert saved["mean"].shape == (8, 20)
        assert saved["covariance_factor"].shape == (8, 20, 20)
        assert (saved["variance"] > 0).all()
        assert not np.allclose(saved["variance"], 1e-4, rtol=1e-6, atol=0)
        a = saved["covariance_factor"]
        covariances = a.transpose(0, 2, 1) @ a
        np.testing.assert_allclose(np.diagonal(covariances, axis1=1, axis2=2), saved["variance"])
        residual = saved["targets"] - saved["mean"]
        expected_nll = (
            20 * np.log(2 * np.pi)
            + np.linalg.slogdet(covariances)[1]
            + np.einsum(
                "bi,bi->b", residual, np.linalg.solve(covariances, residual[..., None])[..., 0]
            )
        ) / 40
        np.testing.assert_allclose(saved["nll_per_asset"], expected_nll, rtol=1e-8)
        np.testing.assert_allclose(saved["weights"].sum(axis=1), 1, atol=1e-6)
        assert (saved["weights"] >= -1e-6).all()
        assert (saved["weights"] <= 0.1 + 1e-6).all()


def test_gaussian_rejects_nonvariance_risk():
    panel = synthetic_panel(periods=10)
    with pytest.raises(ValueError, match="variance risk"):
        run(panel, prediction="gaussian", risk="sd")
