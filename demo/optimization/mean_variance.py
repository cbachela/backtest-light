"""Historical-mean baseline using the same constraints and risk as Wang E2E."""

import argparse

import torch

from btlight.data.panel import load_data
from btlight.estimation.expected_return import mean_arithmetic
from btlight.ml.training.portfolio_data import (
    covariance_factors,
    experiment_splits,
    summarize,
    save_results,
)
from btlight.optimization.differentiable import PortfolioLayer


def run(panel, *, risk_aversion=1.0, first_test="2014-01-01", smoke=False):
    factors = covariance_factors(panel)
    layer = PortfolioLayer(len(panel.assets), risk_aversion)
    previous = torch.full((len(panel.assets),), 1 / len(panel.assets), dtype=torch.float64)
    records = []
    for _, _, _, test in experiment_splits(panel, first_test, factors, smoke):
        with torch.no_grad():
            for t in test:
                # Match the one-year risk lookback; these returns are already observable.
                mu = torch.tensor(
                    mean_arithmetic(panel.risk_returns[t - 251 : t + 1]), dtype=torch.float64
                )
                weights = layer(mu, factors[t], previous)
                records.append((t, weights.numpy(), 0.0))
                previous = weights
    return summarize(panel, records).assign(
        model="Mean-variance", risk="variance", risk_aversion=risk_aversion, cost=0.0, smoke=smoke
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", help="Shared NPZ panel; omit only with --smoke")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--risk-aversion", type=float, nargs="+", default=[1.0])
    parser.add_argument("--output", default="outputs/mean_variance.csv")
    parser.add_argument("--first-test", default="2014-01-01")
    args = parser.parse_args()
    panel = load_data(args.panel, args.smoke)
    results = [
        run(panel, risk_aversion=eta, first_test=args.first_test, smoke=args.smoke)
        for eta in args.risk_aversion
    ]
    save_results(results, args.output, args.panel or "synthetic-seed-7")


if __name__ == "__main__":
    main()
