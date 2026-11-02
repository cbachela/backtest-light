"""Shared point, BL-confidence and single-network Gaussian return predictors."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ReturnPredictor(nn.Module):
    """Wang Appendix A.2.1 backbone; optional confidence head is our extension."""

    def __init__(
        self, features: int, depth: int = 3, *, learned_confidence=False, initial_variance=1e-4
    ):
        super().__init__()
        if features < 1 or depth not in range(6):
            raise ValueError("features must be positive; depth must be 0 (linear) through 5")
        layers = []
        if depth:
            layers.append(nn.LayerNorm(features))
            for width in (32, 16, 8, 4, 2)[:depth]:
                layers.extend(
                    [nn.Linear(features, width), nn.ReLU(), nn.BatchNorm1d(width), nn.Dropout(0.05)]
                )
                features = width
        layers.append(nn.Linear(features, 1))
        self.network = nn.Sequential(*layers)
        self.confidence_head = None
        if learned_confidence:
            if not math.isfinite(initial_variance) or initial_variance <= 0:
                raise ValueError("initial_variance must be finite and positive")
            self.initial_variance = initial_variance
            self.confidence_head = nn.Linear(features, 1)
            # Start with the same Omega as the constant-confidence baseline.
            nn.init.zeros_(self.confidence_head.weight)
            nn.init.zeros_(self.confidence_head.bias)

    def forward(self, features):
        # One complete date cross-section: [assets, features].
        mean, _ = self.predict_views(features)
        return mean

    def predict_views(self, features):
        """Return views and optional diagonal Omega"""
        if self.confidence_head is None:
            return self.network(features).squeeze(-1), None
        hidden = self.network[:-1](features)
        mean = self.network[-1](hidden).squeeze(-1)
        score = self.confidence_head(hidden).squeeze(-1)
        # Relative floor prevents zero variance without detaching the learned head.
        variance = self.initial_variance * (F.softplus(score) + 1e-6) / (math.log(2) + 1e-6)
        return mean, variance


class GaussianReturnPredictor(ReturnPredictor):
    """Single heteroscedastic Gaussian network (Lakshminarayanan et al., 2017).

    Variance describes returns, not BL view confidence or parameter uncertainty.
    Historical correlations are an optional portfolio-specific extension.
    """

    def __init__(self, features, depth=3, *, initial_variance=1e-4, variance_floor=1e-8):
        super().__init__(features, depth)
        if not (math.isfinite(variance_floor) and variance_floor > 0):
            raise ValueError("variance_floor must be finite and positive")
        if not (math.isfinite(initial_variance) and initial_variance > variance_floor):
            raise ValueError("initial_variance must exceed variance_floor and be finite")
        self.initial_variance = initial_variance
        self.variance_floor = variance_floor
        self.variance_head = nn.Linear(self.network[-1].in_features, 1)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.zeros_(self.variance_head.bias)
        # Keep initial mean forecasts in daily-return units.
        nn.init.normal_(self.network[-1].weight, std=1e-3)
        nn.init.zeros_(self.network[-1].bias)

    def predict_moments(self, features):
        """Marginal moments without allocating a dense cross-asset covariance."""
        hidden = self.network[:-1](features)
        mean = self.network[-1](hidden).squeeze(-1)
        score = self.variance_head(hidden).squeeze(-1)
        variance = self.variance_floor + (
            (self.initial_variance - self.variance_floor) * F.softplus(score) / math.log(2)
        )
        return mean, variance

    def predict_distribution(self, features, correlation_cholesky=None):
        """Joint Gaussian for one cross-section; default correlations are identity."""
        mean, variance = self.predict_moments(features)
        if correlation_cholesky is None:
            correlation_cholesky = torch.eye(len(mean), dtype=mean.dtype, device=mean.device)
        if correlation_cholesky.shape != (len(mean), len(mean)):
            raise ValueError("correlation factor must match the asset cross-section")
        # D @ L gives covariance D @ R @ D; transpose for PortfolioLayer.
        scale_tril = variance.sqrt()[:, None] * correlation_cholesky
        return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)


def gaussian_nll(distribution, realized):
    """Joint negative log-likelihood per asset, including the normalizing constant."""
    return -distribution.log_prob(realized) / distribution.event_shape[0]
