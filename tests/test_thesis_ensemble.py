import numpy as np
import pytest
import torch
from btlight.ml.model.ensemble import (
    allocation_factor,
    marginal_quantile,
    mixture_moments,
    mixture_nll,
)
from btlight.optimization.differentiable import PortfolioLayer


def test_mixture_moments_density_and_quantiles():
    means = torch.tensor([[0.1, -0.2], [-0.1, 0.3]], dtype=torch.float64)
    variances = torch.tensor([[0.04, 0.09], [0.01, 0.04]], dtype=torch.float64)
    components = [torch.distributions.MultivariateNormal(m, torch.diag(v))
                  for m, v in zip(means, variances)]
    mean, within, disagreement = mixture_moments(components)
    expected = torch.diag(variances.mean(0)) + (means - mean).T @ (means - mean) / 2
    factor = allocation_factor(components, disagreement)
    torch.testing.assert_close(factor.T @ factor, expected)
    torch.testing.assert_close(within, variances.mean(0))
    realized = torch.tensor([0.2, 0.1], dtype=torch.float64)
    density = sum(d.log_prob(realized).exp() for d in components) / 2
    torch.testing.assert_close(mixture_nll(components, realized), -density.log() / 2)
    for probability in (0.025, 0.975):
        q = marginal_quantile(components, probability)
        cdf = torch.distributions.Normal(means, variances.sqrt()).cdf(q).mean(0)
        torch.testing.assert_close(cdf, torch.full_like(cdf, probability))
    torch.testing.assert_close(mixture_nll(components[:1], realized),
                               -components[0].log_prob(realized) / 2)
    # Extreme observations must not underflow the mixture likelihood.
    assert torch.isfinite(mixture_nll(components, realized * 10000))


def test_low_rank_allocation_matches_dense_and_has_gradients():
    mu = torch.tensor([0.03, 0.02, 0.01], dtype=torch.float64)
    scale = torch.tensor([0.2, 0.3, 0.25], dtype=torch.float64, requires_grad=True)
    disagreement = torch.tensor([[0.1, -0.1, 0.0], [-0.1, 0.1, 0.0]],
                                dtype=torch.float64, requires_grad=True)
    previous = torch.full((3,), 1 / 3, dtype=torch.float64)
    sparse = PortfolioLayer(3, cap=0.9, diagonal=True, disagreement_rank=2)
    dense = PortfolioLayer(3, cap=0.9)
    covariance = torch.diag(scale.square()) + disagreement.T @ disagreement
    reference = dense(mu, torch.linalg.cholesky(covariance).T, previous)
    weights = sparse(mu, scale, previous, disagreement)
    torch.testing.assert_close(weights, reference, atol=5e-5, rtol=2e-4)
    inverse_mu = torch.linalg.solve(covariance, mu)
    inverse_one = torch.linalg.solve(covariance, torch.ones_like(mu))
    multiplier = (inverse_mu.sum() - 2) / inverse_one.sum()
    exact = (inverse_mu - multiplier * inverse_one) / 2
    torch.testing.assert_close(weights, exact, atol=5e-5, rtol=2e-4)
    assert torch.autograd.gradcheck(
        lambda v, u: sparse(mu, v, previous, u), (scale, disagreement),
        eps=1e-3, atol=2e-3, rtol=2e-2,
    )


@pytest.mark.parametrize("finetune", [False, True])
def test_fixed_ensemble_walk_forward(tmp_path, finetune):
    from btlight.data.panel import synthetic_panel
    from btlight.ml.training.e2e import run

    path = tmp_path / "forecasts.npz"
    events = []
    result = run(synthetic_panel(), epochs=1, smoke=True, prediction="gaussian",
                 members=2, risk="variance", finetune=finetune,
                 distribution_output=path, log=lambda event, **kw: events.append((event, kw)))
    assert result.members.iloc[0] == 2
    assert np.isfinite(result.test_nll_per_asset.iloc[0])
    with np.load(path) as saved:
        np.testing.assert_allclose(saved["mean"], saved["member_mean"].mean(1))
        expected = saved["member_variance"].mean(1) + saved["member_mean"].var(1)
        np.testing.assert_allclose(saved["variance"], expected)
        assert not np.allclose(saved["member_mean"][:, 0], saved["member_mean"][:, 1])
        np.testing.assert_array_equal(saved["member_seeds"], [7, 8])
    epochs = [fields for event, fields in events if event == "epoch"]
    assert {r["member"] for r in epochs} == {1, 2}
    assert {r["seed"] for r in epochs} == {7, 8}
