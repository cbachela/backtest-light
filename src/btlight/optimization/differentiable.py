"""Convex portfolio layer retaining gradients through the allocation decision.

This layer chooses long-only, fully invested, capped weights from forecast returns
and variance risk, with an optional turnover penalty. loss() scores those weights using realized returns
and a separately supplied risk factor.
"""

import math

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


class PortfolioLayer(torch.nn.Module):
    def __init__(
        self,
        assets,
        risk_aversion=1.0,
        cap=0.1,
        cost=0.0,
        diagonal=False,
        disagreement_rank=0,
    ):
        super().__init__()
        if not 0 < cap <= 1 or assets * cap < 1:
            raise ValueError("cap must be in (0,1] and assets*cap >= 1")
        if any(not math.isfinite(v) or v < 0 for v in (risk_aversion, cost)):
            raise ValueError("risk aversion and cost must be finite and nonnegative")
        self.risk_aversion, self.cost = risk_aversion, cost
        self.tolerance = 1e-11 if disagreement_rank else 1e-9
        w = cp.Variable(assets)
        mu = cp.Parameter(assets)
        factor = cp.Parameter(assets if diagonal else (assets, assets))
        previous = cp.Parameter(assets)
        # Factor.T @ factor = covariance makes the problem DPP-compliant.
        exposure = cp.multiply(factor, w) if diagonal else factor @ w
        disagreement = None
        if disagreement_rank:
            disagreement = cp.Parameter((disagreement_rank, assets))
            exposure = cp.hstack([exposure, disagreement @ w])
        risk_term = cp.sum_squares(exposure)
        # Minimize negative forecast utility; half the L1 change measures turnover.
        objective = -mu @ w + risk_aversion * risk_term + cost / 2 * cp.norm1(w - previous)
        # Require full investment, no short positions, and a per-asset weight cap.
        problem = cp.Problem(cp.Minimize(objective), [w >= 0, cp.sum(w) == 1, w <= cap])
        parameters = [mu, factor, previous]
        if disagreement is not None:
            parameters.append(disagreement)
        self.solver = CvxpyLayer(problem, parameters=parameters, variables=[w])

    def forward(self, mu, factor, previous, disagreement=None):
        # CvxpyLayer differentiates the solution with respect to its inputs.
        parameters = [mu, factor, previous]
        if disagreement is not None:
            parameters.append(disagreement)
        (weights,) = self.solver(
            *parameters,
            solver_args={
                "solve_method": "Clarabel",
                "tol_gap_abs": self.tolerance,
                "tol_gap_rel": self.tolerance,
                "tol_feas": self.tolerance,
            },
        )
        return weights

    def loss(self, weights, realized, factor, previous):
        exposure = factor * weights if factor.ndim == 1 else factor @ weights
        variance = exposure.square().sum()
        # Wang eq. (21): substitute realized returns in the upper loss.
        return (
            -weights @ realized
            + self.risk_aversion * variance
            + self.cost / 2 * (weights - previous).abs().sum()
        )
