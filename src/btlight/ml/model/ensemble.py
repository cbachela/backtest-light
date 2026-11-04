"""Equal-weight Gaussian mixtures and their portfolio moments."""

import math

import torch
from torch import nn


class GaussianEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        if not members:
            raise ValueError("ensemble requires at least one member")
        self.members = nn.ModuleList(members)

    def predict_components(self, features, correlation=None):
        if correlation is None:
            return [
                torch.distributions.Independent(
                    torch.distributions.Normal(mean, variance.sqrt()), 1
                )
                for mean, variance in (m.predict_moments(features) for m in self.members)
            ]
        return [m.predict_distribution(features, correlation) for m in self.members]


def mixture_moments(components):
    means = torch.stack([d.mean for d in components])
    variances = torch.stack([d.variance for d in components])
    mean = means.mean(0)
    disagreement = (means - mean) / math.sqrt(len(components))
    within = variances.mean(0)
    return mean, within, disagreement


def mixture_nll(components, realized):
    scores = torch.stack([d.log_prob(realized) for d in components])
    return -(torch.logsumexp(scores, 0) - math.log(len(components))) / len(realized)


def marginal_quantile(components, probability):
    means = torch.stack([d.mean for d in components])
    scales = torch.stack([d.stddev for d in components])
    normal = torch.distributions.Normal(means, scales)
    z = means.new_tensor(probability)
    endpoints = normal.icdf(z)
    lower, upper = endpoints.min(0).values, endpoints.max(0).values
    for _ in range(50):
        middle = (lower + upper) / 2
        below = normal.cdf(middle).mean(0) < probability
        lower = torch.where(below, middle, lower)
        upper = torch.where(below, upper, middle)
    return (lower + upper) / 2


def allocation_factor(components, disagreement):
    covariance = torch.stack([d.covariance_matrix for d in components]).mean(0)
    covariance = covariance + disagreement.T @ disagreement
    return torch.linalg.cholesky(covariance).T
