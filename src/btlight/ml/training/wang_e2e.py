"""Wang point-estimate baseline with MSE pretraining and optional fixed-Omega BL."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from btlight.data.french import read_daily_table
from btlight.estimation.differentiable_bl import BLUpdate, historical_bl_inputs, load_market_weights
from btlight.estimation.prediction_errors import historical_errors, predict_history
from btlight.ml.model.return_predictor import ReturnPredictor
from btlight.ml.training.portfolio_data import covariance_factors, experiment_splits, summarize
from btlight.optimization.differentiable import PortfolioLayer


def fit(
    model,
    panel,
    train,
    validation,
    factors,
    layer,
    epochs,
    mode,
    bl=None,
    priors=None,
    bl_covariances=None,
):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if mode not in {"mse", "e2e"}:
        raise ValueError("mode must be mse or e2e")
    if not len(train) or not len(validation):
        raise ValueError("empty training/validation window")
    # Appendix A.2.2: Adam, eight dates/update, patience 10, MSE warm start.
    learning_rate = 1e-4 if layer.risk_aversion < 1 else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x = torch.tensor(panel.features, dtype=torch.float64)
    y = torch.tensor(panel.targets, dtype=torch.float64)
    n = panel.targets.shape[1]
    history, best, best_state, stale = [], float("inf"), None, 0

    def run(indices, training):
        model.train(training)
        previous = torch.full((n,), 1 / n, dtype=torch.float64)
        total = 0.0
        for offset in range(0, len(indices), 8):
            batch = indices[offset : offset + 8]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for t in batch:
                predicted = model(x[t])
                if mode == "mse":
                    loss = (predicted - y[t]).square().mean()
                else:
                    if bl is not None:
                        covariance = bl_covariances[t]
                        predicted = bl(predicted, priors[t], covariance)
                    weights = layer(predicted, factors[t], previous)
                    loss = layer.loss(weights, y[t], factors[t], previous)
                    # Preserve the temporal derivative within the consecutive block.
                    previous = weights
                losses.append(loss)
            objective = torch.stack(losses).mean()
            if not torch.isfinite(objective):
                raise RuntimeError("nonfinite training/validation loss")
            if training:
                objective.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise RuntimeError("nonfinite gradient")
                optimizer.step()
            total += objective.item() * len(batch)
            # Explicit truncated backpropagation boundary; holdings persist.
            previous = previous.detach()
        return total / len(indices)

    for epoch in range(epochs):
        train_loss = run(train, True)
        with torch.no_grad():
            val_loss = run(validation, False)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best:
            best, best_state, stale = val_loss, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    model.eval()
    return history


def run(
    panel,
    *,
    epochs=100,
    risk_aversion=1.0,
    risk="sd",
    cost=0.0,
    first_test="2014-01-01",
    smoke=False,
    uncertainty_output=None,
    error_window=504,
    bl_mode="off",
    bl_variance=1e-4,
    bl_tau=0.05,
    bl_factors=None,
    bl_market_weights=None,
    bl_delta=2.5,
):
    if bl_mode not in {"off", "constant", "historical"}:
        raise ValueError("bl_mode must be off, constant or historical")
    if bl_mode != "off":
        BLUpdate(torch.full((len(panel.assets),), bl_variance), tau=bl_tau)
        if bl_mode == "historical" and error_window < 2:
            raise ValueError("error_window must be at least 2")
    torch.manual_seed(7)
    torch.set_num_threads(1)
    factors = covariance_factors(panel)
    priors, bl_covariances = None, None
    if bl_mode != "off":
        if smoke:
            # Synthetic excess targets have zero unconditional mean; no market file exists.
            priors = {t: torch.zeros(len(panel.assets), dtype=torch.float64) for t in factors}
            bl_covariances = {t: factor.T @ factor for t, factor in factors.items()}
        else:
            if bl_factors is None or bl_market_weights is None:
                raise ValueError("BL requires --bl-factors and --bl-market-weights")
            market_weights = load_market_weights(bl_market_weights, panel.assets)
            priors, bl_covariances = historical_bl_inputs(
                panel, market_weights, read_daily_table(bl_factors, ",Mkt-RF"), delta=bl_delta
            )
            factors = {t: factor for t, factor in factors.items() if t in priors}
    layer = PortfolioLayer(len(panel.assets), risk_aversion, risk=risk, cost=cost)
    records = []
    uncertainty_records = []
    if uncertainty_output is not None and error_window < 2:
        raise ValueError("error_window must be at least 2")
    previous = torch.full((len(panel.assets),), 1 / len(panel.assets), dtype=torch.float64)
    for quarter, train, val, test in experiment_splits(panel, first_test, factors, smoke):
        if uncertainty_output is not None:
            available = (panel.label_end <= panel.dates[test[0]]).sum()
            if available < error_window:
                raise ValueError(
                    f"Need {error_window} observed errors before {quarter}; found {available}"
                )
        model = ReturnPredictor(panel.features.shape[-1]).double()
        print(f"{quarter}: MSE pretraining, then E2E (BL={bl_mode})", flush=True)
        if bl_mode == "historical":
            available = (panel.label_end <= panel.dates[train[-1]]).sum()
            if available < error_window:
                raise ValueError("Not enough observed training history for fixed residual variance")
        fit(model, panel, train, val, factors, layer, epochs, "mse")
        bl = None
        if bl_mode != "off":
            variance = torch.full((len(panel.assets),), bl_variance, dtype=torch.float64)
            if bl_mode == "historical":
                # Estimate only from training history, then freeze throughout E2E/validation/test.
                with torch.no_grad():
                    predictions = predict_history(model, panel, train[-1] + 1)
                    estimate = historical_errors(panel, predictions, train[-1], error_window)
                    variance = estimate.variance.clamp_min(1e-8)
            bl = BLUpdate(variance, tau=bl_tau)
        fit(
            model,
            panel,
            train,
            val,
            factors,
            layer,
            epochs,
            "e2e",
            bl=bl,
            priors=priors,
            bl_covariances=bl_covariances,
        )
        with torch.no_grad():
            if uncertainty_output is not None:
                # Reuse this quarter's fitted model; targets are filtered by availability below.
                predictions = predict_history(model, panel, test[-1] + 1)
            for t in test:
                mu = model(torch.tensor(panel.features[t], dtype=torch.float64))
                allocation_mean = mu
                if bl is not None:
                    allocation_mean = bl(mu, priors[t], bl_covariances[t])
                weights = layer(allocation_mean, factors[t], previous)
                if uncertainty_output is not None:
                    estimate = historical_errors(panel, predictions, t, error_window)
                    uncertainty_records.append(
                        (t, mu.numpy(), estimate.mean_error.numpy(), estimate.covariance.numpy())
                    )
                turnover = (weights - previous).abs().sum().item() / 2
                records.append((t, weights.numpy(), cost * turnover))
                previous = weights
    if uncertainty_output is not None:
        path = Path(uncertainty_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = [record[0] for record in uncertainty_records]
        covariance = np.stack([record[3] for record in uncertainty_records])
        np.savez_compressed(
            path,
            dates=panel.dates[ids],
            label_end=panel.label_end[ids],
            assets=panel.assets,
            predictions=np.stack([record[1] for record in uncertainty_records]),
            mean_error=np.stack([record[2] for record in uncertainty_records]),
            covariance=covariance,
            variance=np.diagonal(covariance, axis1=1, axis2=2),
            lookback=error_window,
            estimator="current-model-centered-residuals-ddof0",
        )
        print(f"Saved historical-error estimates to {path}")
    model_name = "Wang E2E" if bl_mode == "off" else f"Wang E2E + BL ({bl_mode})"
    result = summarize(panel, records).assign(
        model=model_name, risk=risk, risk_aversion=risk_aversion, cost=cost, smoke=smoke
    )
    if bl_mode != "off":
        result = result.assign(
            bl_mode=bl_mode,
            bl_tau=bl_tau,
            bl_prior="zero-synthetic" if smoke else "reverse-CAPM",
            bl_delta=bl_delta,
            bl_error_window=error_window if bl_mode == "historical" else 0,
        )
        if bl_mode == "constant":
            result = result.assign(bl_variance=bl_variance)
    return result
