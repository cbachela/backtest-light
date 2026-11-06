"""Walk through point-prediction pretraining and one E2E update.

Run from the backtest-light repository root with:
    uv run --no-sync python demo/ml/point_e2e_demo.py
    
This uses one synthetic cross-section (8 assets, 10 features). 
The production  runner uses rolling chronological windows: 39 months for training,
9 for validation, then 3 for testing, advancing by one quarter. Model is retrained 
every quarter. The demo shows one training step and one E2E step on a single date.

See readme for the list of files that are used in the implementation.
"""

import torch
from btlight.data.panel import synthetic_panel
from btlight.ml.model.return_predictor import ReturnPredictor
from btlight.ml.training.portfolio_data import covariance_factors
from btlight.optimization.differentiable import PortfolioLayer


def main():
    torch.manual_seed(7)
    panel = synthetic_panel(seed=7, periods=300, assets=8, features=10)
    assets = len(panel.assets)
    features = panel.features.shape[-1]
    date_index = 251  # First date with a full 252-date historical risk window.

    # features and target
    x = torch.tensor(panel.features[date_index], dtype=torch.float64)
    y = torch.tensor(panel.targets[date_index], dtype=torch.float64)

    # Portfolio risk is return covariance across the assets. The factor
    # A satisfies A.T @ A = covariance and comes from the trailing 252 known
    # risk-return observations.
    risk_factor = covariance_factors(panel, lookback=252)[date_index]
    print(
        f"Panel date {panel.dates[date_index]} (label ends {panel.label_end[date_index]}): "
        f"{assets} assets, {x.shape[1]} features"
    )

    # Here depth=0 so it's a linear network. Production uses depth=3,
    # see src/btlight/ml/training/e2e.py for full network specification.
    model = ReturnPredictor(features, depth=0).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # PTO: fit forecasts directly to observed returns with mean squared error.
    point_forecast = model(x)
    mse = (point_forecast - y).square().mean()
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()
    print(f"PTO step: mean squared error = {mse.item():.6f}")

    # Start from equal holdings. Production carries prior test-period holdings
    # forward; this single-date illustration has no previous rebalance to carry.
    previous_weights = torch.full((assets,), 1 / assets, dtype=torch.float64)
    allocator = PortfolioLayer(assets, risk_aversion=1.0, cap=0.25, cost=0.001)

    # E2E: the allocator chooses weights from forecasts and covariance risk.
    # The loss scores those weights using realized returns, the same risk factor,
    # and turnover from previous holdings. Autograd differentiates through the
    # optimization solution, so this loss updates the return predictor.
    optimizer.zero_grad()
    point_forecast = model(x)
    weights = allocator(point_forecast, risk_factor, previous_weights)
    decision_loss = allocator.loss(weights, y, risk_factor, previous_weights)
    decision_loss.backward()
    gradient_norm = torch.linalg.vector_norm(
        torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
    )
    optimizer.step()

    print(f"E2E step: realized decision loss = {decision_loss.item():.6f}")
    print(f"Gradient norm reaching predictor = {gradient_norm.item():.6f}")
    print("Allocated weights:", weights.detach().numpy().round(3))
    print("Weights sum to:", weights.detach().sum().item())


if __name__ == "__main__":
    main()
