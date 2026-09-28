"""Convex portfolio layer retaining gradients through the allocation decision.

The existing NumPy/qpsolvers interface is suitable for backtesting, but cannot
propagate PyTorch gradients during end-to-end training.
"""

import math

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


class PortfolioLayer(torch.nn.Module):
    def __init__(self, assets, risk_aversion=1.0, cap=0.1, risk="variance", cost=0.0):
        super().__init__()
        if risk not in {"variance", "sd"}:
            raise ValueError("risk must be variance or sd")
        if not 0 < cap <= 1 or assets * cap < 1:
            raise ValueError("cap must be in (0,1] and assets*cap >= 1")
        if any(not math.isfinite(v) or v < 0 for v in (risk_aversion, cost)):
            raise ValueError("risk aversion and cost must be finite and nonnegative")
        self.risk_aversion, self.risk, self.cost = risk_aversion, risk, cost
        w = cp.Variable(assets)
        mu = cp.Parameter(assets)
        factor = cp.Parameter((assets, assets))
        previous = cp.Parameter(assets)
        # Factor.T @ factor = covariance makes the problem DPP-compliant.
        risk_term = cp.sum_squares(factor @ w) if risk == "variance" else cp.norm(factor @ w)
        objective = -mu @ w + risk_aversion * risk_term + cost / 2 * cp.norm1(w - previous)
        problem = cp.Problem(cp.Minimize(objective), [w >= 0, cp.sum(w) == 1, w <= cap])
        self.solver = CvxpyLayer(problem, parameters=[mu, factor, previous], variables=[w])

    def forward(self, mu, factor, previous):
        (weights,) = self.solver(
            mu,
            factor,
            previous,
            solver_args={
                "solve_method": "Clarabel",
                "tol_gap_abs": 1e-9,
                "tol_gap_rel": 1e-9,
                "tol_feas": 1e-9,
            },
        )
        return weights

    def loss(self, weights, realized, factor, previous):
        exposure = factor @ weights
        risk = exposure.square().sum() if self.risk == "variance" else exposure.norm()
        # Wang eq. (21): substitute realized returns in the upper loss.
        return (
            -weights @ realized
            + self.risk_aversion * risk
            + self.cost / 2 * (weights - previous).abs().sum()
        )
