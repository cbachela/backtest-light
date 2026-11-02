"""Shared walk-forward training for point and Gaussian portfolio models."""

import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from btlight.data.panel import DynamicPanel
from btlight.ml.model.return_predictor import (
    GaussianReturnPredictor,
    ReturnPredictor,
    gaussian_nll,
)
from btlight.ml.training.portfolio_data import (
    align_holdings,
    correlation_factors,
    covariance_factors,
    experiment_splits,
    summarize,
)
from btlight.optimization.differentiable import PortfolioLayer


def dynamic_distribution(model, features):
    mean, variance = model.predict_moments(features)
    return torch.distributions.Independent(torch.distributions.Normal(mean, variance.sqrt()), 1)


def fit(
    model,
    panel,
    train,
    validation,
    factors,
    layer,
    epochs,
    mode,
    correlations=None,
    nll_weight=1e-3,
    executed_exits=None,
    log=None,
    quarter=None,
):
    """Warm start or fine-tune the shared network on date cross-sections.

    Dynamic dates must have complete, aligned labels. No future-label filtering
    or missing-return substitution is allowed in a portfolio objective.
    """
    dynamic = isinstance(panel, DynamicPanel)
    if dynamic:
        panel.validate_portfolio_dates(train)
        panel.validate_portfolio_dates(validation)

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if mode not in {"mse", "nll", "e2e"}:
        raise ValueError("mode must be mse, nll or e2e")
    probabilistic = isinstance(model, GaussianReturnPredictor)
    if probabilistic:
        if (correlations is None and not dynamic) or mode == "mse":
            raise ValueError("Gaussian training requires correlations and nll/e2e")
        if not math.isfinite(nll_weight) or nll_weight <= 0:
            raise ValueError("nll_weight must be finite and positive")
    elif mode == "nll":
        raise ValueError("nll requires a Gaussian predictor")
    if not len(train) or not len(validation):
        raise ValueError("empty training/validation window")

    # Wang-style schedule: Adam, eight dates/update, patience 10.
    learning_rate = 1e-4 if layer.risk_aversion < 1 else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_dates, patience = 8, 10
    if log:
        log(
            "stage_start",
            quarter=quarter,
            stage=mode,
            learning_rate=learning_rate,
            optimizer="Adam",
            batch_dates=batch_dates,
            patience=patience,
            max_epochs=epochs,
        )

    # Only materialize each date; the European packed panel has millions of rows.
    @lru_cache(maxsize=4)
    def allocation_layer(n):
        return PortfolioLayer(
            n, layer.risk_aversion, risk=layer.risk, cost=layer.cost, diagonal=True
        )

    def cross_section(t):
        sl = panel.day(t) if dynamic else t
        x = torch.as_tensor(panel.features[sl], dtype=torch.float64)
        y = torch.as_tensor(panel.targets[sl], dtype=torch.float64)
        factor = (
            torch.as_tensor(panel.risk_variance[sl], dtype=torch.float64).sqrt()
            if dynamic
            else factors[t]
        )
        return x, y, factor

    history, best, best_state, stale = [], float("inf"), None, 0

    def run(indices, training):
        model.train(training)
        previous = None
        previous_ids = []
        total = 0.0
        for offset in range(0, len(indices), batch_dates):
            batch = indices[offset : offset + batch_dates]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for t in batch:
                x, y, factor = cross_section(t)
                n = len(y)
                distribution = None
                if probabilistic:
                    distribution = (
                        dynamic_distribution(model, x)
                        if dynamic
                        else model.predict_distribution(x, correlations[t])
                    )
                    predicted = distribution.mean
                    nll = gaussian_nll(distribution, y)
                else:
                    predicted = model(x)

                # Pretraining never constructs or solves an allocation problem.
                if mode != "e2e":
                    losses.append(nll if probabilistic else (predicted - y).square().mean())
                    continue

                allocator = allocation_layer(n) if dynamic else layer
                exit_weight = x.new_zeros(())
                if dynamic:
                    current_ids = panel.asset_index[panel.day(t)].tolist()
                    if previous is None:
                        previous = x.new_zeros(n)
                    else:
                        previous, exit_weight = align_holdings(
                            previous_ids,
                            previous,
                            current_ids,
                            () if executed_exits is None else executed_exits.get(t, ()),
                        )
                    previous_ids = current_ids
                elif previous is None:
                    previous = torch.full((n,), 1 / n, dtype=torch.float64)

                allocation_factor = factor
                if distribution is not None:
                    allocation_factor = (
                        distribution.stddev if dynamic else distribution.scale_tril.T
                    )

                weights = allocator(predicted, allocation_factor, previous)
                # Score risk independently of the covariance predicted for allocation.
                loss = allocator.loss(weights, y, factor, previous)
                loss = loss + allocator.cost / 2 * exit_weight
                if probabilistic:
                    loss = loss + nll_weight * nll
                losses.append(loss)
                previous = weights

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
            if previous is not None:
                previous = previous.detach()
        return total / len(indices)

    for epoch in range(epochs):
        started = perf_counter()
        if log:
            log("epoch_start", quarter=quarter, stage=mode, epoch=epoch + 1)
        train_loss = run(train, True)
        train_seconds = perf_counter() - started
        with torch.no_grad():
            val_loss = run(validation, False)
        epoch_seconds = perf_counter() - started
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best:
            best, best_state, stale = val_loss, deepcopy(model.state_dict()), 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if log:
            log(
                "epoch",
                quarter=quarter,
                stage=mode,
                epoch=epoch + 1,
                train_loss=train_loss,
                validation_loss=val_loss,
                epoch_seconds=epoch_seconds,
                train_seconds=train_seconds,
                validation_seconds=epoch_seconds - train_seconds,
                best_epoch=best_epoch,
                best_validation_loss=best,
            )
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    if log:
        log(
            "stage_end",
            quarter=quarter,
            stage=mode,
            epochs_completed=len(history),
            best_epoch=best_epoch,
            best_validation_loss=best,
            early_stopped=stale >= patience,
        )
    return history


def run(
    panel,
    *,
    epochs=100,
    pretrain=True,
    risk_aversion=1.0,
    risk="sd",
    cost=0.0,
    first_test="2014-01-01",
    smoke=False,
    prediction="point",
    initial_variance=1e-4,
    variance_floor=1e-8,
    correlation_shrinkage=0.1,
    nll_weight=1e-3,
    distribution_output=None,
    executed_exits=None,
    log=None,
):
    dynamic = isinstance(panel, DynamicPanel)
    if prediction not in {"point", "gaussian"}:
        raise ValueError("prediction must be point or gaussian")
    probabilistic = prediction == "gaussian"
    if probabilistic:
        if risk != "variance":
            raise ValueError("Gaussian mode requires variance risk")
        if not math.isfinite(nll_weight) or nll_weight <= 0:
            raise ValueError("nll_weight must be finite and positive")
    if distribution_output is not None:
        if not probabilistic or Path(distribution_output).suffix != ".npz":
            raise ValueError("distribution_output requires gaussian prediction and an .npz path")

    if log:
        log(
            "settings",
            prediction=prediction,
            pretrain=pretrain,
            max_epochs=epochs,
            risk=risk,
            risk_aversion=risk_aversion,
            cost=cost,
            first_test=first_test,
            initial_variance=initial_variance if probabilistic else None,
            variance_floor=variance_floor if probabilistic else None,
            nll_weight=nll_weight if probabilistic else None,
            correlation_shrinkage=correlation_shrinkage if probabilistic and not dynamic else 0.0,
            covariance_model=(
                "historical-correlation" if probabilistic and not dynamic else "diagonal"
            ),
            asset_cap=0.1,
            seed=7,
            dtype="float64",
            torch_threads=1,
            solver="Clarabel",
            dynamic=dynamic,
            smoke=smoke,
            features=int(panel.features.shape[-1]),
            assets=len(panel.assets),
            dates=len(panel.dates),
            data_start=str(panel.dates[0]),
            data_end=str(panel.dates[-1]),
            train_months=39,
            validation_months=9,
            test_months=3,
        )

    torch.manual_seed(7)
    torch.set_num_threads(1)
    factors = range(len(panel.dates)) if dynamic else covariance_factors(panel)
    correlations = (
        correlation_factors(panel, shrinkage=correlation_shrinkage)
        if probabilistic and not dynamic
        else None
    )

    @lru_cache(maxsize=4)
    def allocation_layer(n):
        return PortfolioLayer(n, risk_aversion, risk=risk, cost=cost, diagonal=dynamic)

    layer = allocation_layer(int(panel.offsets[1]) if dynamic else len(panel.assets))
    records = []
    distribution_records = []
    forecast_scores = []
    previous = (
        torch.zeros(0, dtype=torch.float64)
        if dynamic
        else torch.full((len(panel.assets),), 1 / len(panel.assets), dtype=torch.float64)
    )
    previous_ids, previous_end = [], None
    for quarter, train, val, test in experiment_splits(panel, first_test, factors, smoke):
        if dynamic:
            for ids in (train, val, test):
                panel.validate_portfolio_dates(ids)
        if probabilistic:
            model = GaussianReturnPredictor(
                panel.features.shape[-1],
                initial_variance=initial_variance,
                variance_floor=variance_floor,
            ).double()
        else:
            model = ReturnPredictor(panel.features.shape[-1]).double()
        if log:
            log(
                "window_start",
                quarter=quarter,
                architecture=str(model),
                splits={
                    name: {
                        "dates": len(ids),
                        "start": str(panel.dates[ids[0]]),
                        "end": str(panel.dates[ids[-1]]),
                    }
                    for name, ids in (("train", train), ("validation", val), ("test", test))
                },
            )
        pretraining = "nll" if probabilistic else "mse"
        schedule = (
            f"{pretraining.upper()} pretraining, then E2E"
            if pretrain
            else "E2E from initialization"
        )
        print(
            f"{quarter}: {schedule} ({prediction})",
            flush=True,
        )
        if pretrain:
            fit(
                model,
                panel,
                train,
                val,
                factors,
                layer,
                epochs,
                pretraining,
                correlations=correlations,
                nll_weight=nll_weight,
                executed_exits=executed_exits,
                log=log,
                quarter=quarter,
            )
        fit(
            model,
            panel,
            train,
            val,
            factors,
            layer,
            epochs,
            "e2e",
            correlations=correlations,
            nll_weight=nll_weight,
            executed_exits=executed_exits,
            log=log,
            quarter=quarter,
        )

        if log:
            log("test_start", quarter=quarter, dates=len(test))
        with torch.no_grad():
            for t in test:
                sl = panel.day(t) if dynamic else t
                features = torch.tensor(panel.features[sl], dtype=torch.float64)
                exit_weight = features.new_zeros(())
                allocator = layer
                if dynamic:
                    if previous_end is not None and panel.entry_dates[sl][0] != previous_end:
                        raise ValueError("Nonconsecutive execution between test windows")
                    previous_end = panel.label_end[sl][0]
                    current_ids = panel.asset_index[sl].tolist()
                    previous, exit_weight = align_holdings(
                        previous_ids,
                        previous,
                        current_ids,
                        () if executed_exits is None else executed_exits.get(t, ()),
                    )
                    previous_ids = current_ids
                    allocator = allocation_layer(len(current_ids))

                if probabilistic:
                    distribution = (
                        dynamic_distribution(model, features)
                        if dynamic
                        else model.predict_distribution(features, correlations[t])
                    )
                    mu = distribution.mean
                    allocation_factor = (
                        distribution.stddev if dynamic else distribution.scale_tril.T
                    )
                else:
                    mu = model(features)
                    allocation_factor = (
                        torch.as_tensor(panel.risk_variance[sl], dtype=torch.float64).sqrt()
                        if dynamic
                        else factors[t]
                    )

                weights = allocator(mu, allocation_factor, previous)
                if probabilistic:
                    realized = torch.tensor(panel.targets[sl], dtype=torch.float64)
                    nll = gaussian_nll(distribution, realized).item()
                    standardized = (realized - mu) / distribution.stddev
                    coverage = (standardized.abs() <= 1.959963984540054).double().mean().item()
                    forecast_scores.append((nll, coverage))
                    if distribution_output is not None:
                        distribution_records.append(
                            (
                                t,
                                mu.numpy(),
                                distribution.variance.numpy(),
                                allocation_factor.numpy(),
                                weights.numpy(),
                            )
                        )

                turnover = ((weights - previous).abs().sum() + exit_weight).item() / 2
                records.append((t, weights.numpy(), cost * turnover))
                previous = weights

        if log:
            log("window_end", quarter=quarter, test_dates=len(test))

    if distribution_output is not None and dynamic:
        path = Path(distribution_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = [r[0] for r in distribution_records]
        np.savez_compressed(
            path,
            dates=panel.dates[ids],
            offsets=np.r_[0, np.cumsum([len(r[1]) for r in distribution_records])],
            asset_index=np.concatenate([panel.asset_index[panel.day(t)] for t in ids]),
            assets=panel.assets,
            mean=np.concatenate([r[1] for r in distribution_records]),
            variance=np.concatenate([r[2] for r in distribution_records]),
            weights=np.concatenate([r[4] for r in distribution_records]),
            targets=np.concatenate([panel.targets[panel.day(t)] for t in ids]),
            interpretation="dynamic Gaussian E2E; diagonal covariance; confirmed exits only",
        )
    elif distribution_output is not None:
        path = Path(distribution_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = [record[0] for record in distribution_records]
        np.savez_compressed(
            path,
            dates=panel.dates[ids],
            label_end=panel.label_end[ids],
            assets=panel.assets,
            mean=np.stack([r[1] for r in distribution_records]),
            variance=np.stack([r[2] for r in distribution_records]),
            covariance_factor=np.stack([r[3] for r in distribution_records]),
            weights=np.stack([r[4] for r in distribution_records]),
            targets=panel.targets[ids],
            nll_per_asset=np.array(forecast_scores)[:, 0],
            correlation_shrinkage=correlation_shrinkage,
            nll_weight=nll_weight,
            initial_variance=initial_variance,
            variance_floor=variance_floor,
            interpretation="conditional Gaussian panel-target returns; no parameter posterior",
        )
        print(f"Saved return-distribution diagnostics to {path}")

    model_name = "Wang E2E"
    if probabilistic:
        model_name = "Gaussian returns E2E"
    result = summarize(panel, records).assign(
        model=model_name,
        risk=risk,
        risk_aversion=risk_aversion,
        cost=cost,
        smoke=smoke,
        pretrain=pretrain,
    )
    if probabilistic:
        result = result.assign(
            prediction=prediction,
            nll_weight=nll_weight,
            initial_variance=initial_variance,
            variance_floor=variance_floor,
            correlation_shrinkage=0.0 if dynamic else correlation_shrinkage,
            covariance_model="diagonal" if dynamic else "historical-correlation",
            test_nll_per_asset=np.mean(forecast_scores, axis=0)[0],
            marginal_coverage_95=np.mean(forecast_scores, axis=0)[1],
        )
    return result
