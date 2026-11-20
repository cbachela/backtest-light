import pytest

pytest.importorskip("torch")
pytest.importorskip("cvxpylayers")

import numpy as np
import torch
from btlight.data.panel import synthetic_panel
from btlight.ml.training.portfolio_data import rolling_splits, covariance_factors
from btlight.optimization.differentiable import PortfolioLayer
from btlight.ml.training.e2e import run


def test_paper_example_and_gradient():
    x = torch.tensor([[0.3, 0.1], [0.35, 0.2], [0.1, 0.9], [0.2, 1.0]], dtype=torch.float64)
    realized = torch.tensor([0.32, 0.39, 0.92, 1.04], dtype=torch.float64)
    covariance = torch.tensor(
        [
            [0.0064, 0.00288, 0, 0],
            [0.00288, 0.0144, 0, 0.0084],
            [0, 0, 0.36, 0.294],
            [0, 0.0084, 0.294, 0.49],
        ],
        dtype=torch.float64,
    )
    factor = torch.linalg.cholesky(covariance).T
    previous = torch.full((4,), 0.25, dtype=torch.float64)
    # Wang Section 3.1, Table 1: independently reported utilities.
    for eta, expected in [(10, [0.3197, 0.3201, 0.3190]), (1, [0.6284, 0.6253, 0.6290])]:
        layer = PortfolioLayer(4, eta, cap=1.0)
        utilities = []
        for tilt in ([1, 1], [1.05, 0.95], [0.95, 1.05]):
            mu = x @ (torch.tensor([0.652, 0.927], dtype=torch.float64) * torch.tensor(tilt))
            weights = layer(mu, factor, previous)
            utilities.append(-layer.loss(weights, realized, factor, previous).item())
        np.testing.assert_allclose(utilities, expected, atol=1e-4)
    layer = PortfolioLayer(3, cap=0.8)
    mu = torch.tensor([0.02, 0.03, 0.015], dtype=torch.float64, requires_grad=True)
    factor = torch.eye(3, dtype=torch.float64) * 0.3
    previous = torch.ones(3, dtype=torch.float64) / 3
    assert torch.autograd.gradcheck(
        lambda m: layer(m, factor, previous), (mu,), eps=1e-4, atol=2e-3, rtol=2e-2
    )
    weights = layer(mu, factor, previous)
    assert weights.min() >= -1e-7 and weights.max() <= 0.8 + 1e-7
    assert abs(weights.sum().item() - 1) < 1e-7


def test_no_future_information():
    panel = synthetic_panel()
    _, train, val, _ = next(rolling_splits(panel, "2014-01-01"))
    assert (panel.label_end[train] < np.datetime64("2013-04-01")).all()
    assert (panel.label_end[val] < np.datetime64("2014-01-01")).all()
    before = covariance_factors(panel)[300]
    panel.risk_returns[301:] = 100
    assert torch.equal(before, covariance_factors(panel)[300])


def test_expanding_splits():
    panel = synthetic_panel()
    splits = list(rolling_splits(
        panel, "2013-01-01", train_months=None, validation_months=12, test_months=12
    ))
    (_, train, val, test), (_, later_train, _, _) = splits[:2]
    assert train[0] == later_train[0] == 0 and len(later_train) > len(train)
    assert (panel.label_end[train] < np.datetime64("2012-01-01")).all()
    assert (panel.label_end[val] < np.datetime64("2013-01-01")).all()
    assert panel.dates[test[0]] >= np.datetime64("2013-01-01")
    assert panel.dates[test[-1]] < np.datetime64("2014-01-01")


def test_e2e_runs():
    result = run(synthetic_panel(), epochs=1, smoke=True)
    assert result.observations.iloc[0] == 8
    # Long-only, fully invested and capped at 10%: at least ten effective positions.
    assert 10 - 1e-6 <= result.effective_n.iloc[0] and result.max_weight.iloc[0] <= 0.1 + 1e-6
    assert result.annualized_excess_return_net_10bp.iloc[0] == pytest.approx(
        result.annualized_excess_return.iloc[0] - 252 * 1e-3 * result.turnover.iloc[0]
    )
    returns = result.attrs["returns"]
    assert result.ew_active_return.iloc[0] == pytest.approx(
        (returns.portfolio - returns.equal_weight).mean() * 252
    )
    assert np.isfinite(result.select_dtypes(include="number").to_numpy()).all()


def test_performance_and_relative_metrics():
    from btlight.ml.training.portfolio_data import performance, relative

    r = np.array([0.1, -0.05, 0.02, 0.03])
    stats = performance(r, 12)
    assert stats["annualized_excess_return"] == pytest.approx(r.mean() * 12)
    assert stats["cagr"] == pytest.approx(np.prod(1 + r) ** 3 - 1)
    assert stats["sharpe_ratio"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(12))
    assert stats["max_drawdown_excess"] == pytest.approx(0.05)
    assert stats["hit_rate"] == 0.75 and stats["worst_period"] == -0.05
    # A leveraged copy of the benchmark: beta 2, zero alpha; active = benchmark itself.
    b = r / 2
    rel = relative(r, b, 12, "ew_")
    assert rel["ew_beta"] == pytest.approx(2) and rel["ew_alpha"] == pytest.approx(0, abs=1e-12)
    assert rel["ew_tracking_error"] == pytest.approx(b.std(ddof=1) * np.sqrt(12))
    assert rel["ew_information_ratio"] == pytest.approx(b.mean() / b.std(ddof=1) * np.sqrt(12))
    assert rel["ew_cumulative_relative"] == pytest.approx(np.prod(1 + r) / np.prod(1 + b) - 1)
    assert rel["ew_hit_rate"] == 0.75


def test_historical_errors_timing_covariance_and_gradient():
    from btlight.estimation.prediction_errors import historical_errors

    panel = synthetic_panel(periods=40, assets=3)
    predictions = torch.tensor(panel.features[:, :, 0], dtype=torch.float64)
    estimate = historical_errors(panel, predictions, t=15, lookback=5)
    # The target at index 13 becomes observable at signal 15; index 14 does not.
    np.testing.assert_array_equal(estimate.indices, np.arange(9, 14))
    errors = panel.targets[9:14] - predictions[9:14].numpy()
    np.testing.assert_allclose(estimate.covariance, np.cov(errors, rowvar=False, ddof=0))
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    projected = estimate.residuals @ weights
    torch.testing.assert_close(
        weights @ estimate.covariance @ weights, projected.var(unbiased=False)
    )

    panel.targets[14:] = 999
    predictions[14:] = -999
    after = historical_errors(panel, predictions, t=15, lookback=5)
    torch.testing.assert_close(after.covariance, estimate.covariance)
    with pytest.raises(ValueError, match="observed errors"):
        historical_errors(panel, predictions, t=3, lookback=5)

    scale = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

    def covariance_for_scale(value):
        return historical_errors(panel, predictions * value, 15, 5).covariance

    assert torch.autograd.gradcheck(covariance_for_scale, (scale,))
    assert covariance_for_scale(scale).sum().requires_grad


def test_error_predictions_preserve_model_state_and_gradients():
    from btlight.estimation.prediction_errors import predict_history
    from btlight.ml.model.return_predictor import ReturnPredictor

    panel = synthetic_panel(periods=10, assets=12)
    model = ReturnPredictor(6).double().train()
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    predictions = predict_history(model, panel, 5)
    repeated = predict_history(model, panel, 5)
    torch.testing.assert_close(predictions, repeated)
    assert model.training
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name])
    predictions.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_equal_weight_returns_without_training_or_solver(monkeypatch):
    import btlight.ml.training.e2e as training

    def unexpected(*args, **kwargs):
        raise AssertionError("Equal weight must not train or solve")

    monkeypatch.setattr(training, "fit", unexpected)
    monkeypatch.setattr(training, "PortfolioLayer", unexpected)
    panel = synthetic_panel()
    _, _, _, test = next(training.experiment_splits(
        panel, "2014-01-01", covariance_factors(panel), smoke=True
    ))
    result = run(panel, prediction="equal_weight", smoke=True, cost=0.001)
    returns = panel.raw_returns[test].mean(axis=1)
    assert result.annualized_excess_return.iloc[0] == pytest.approx(returns.mean() * 252)
