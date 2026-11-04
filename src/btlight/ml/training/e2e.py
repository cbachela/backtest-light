"""Train and evaluate portfolios in quarterly walk-forward windows.

fit() handles forecast pretraining and E2E learning through the allocation layer.
run() fits a fresh model per window, evaluates held-out dates, and summarizes
performance. It also supports forecast-only training (PTO) and equal weights.
Fixed panels can use historical correlations, changing universes use diagonal risk.
"""

import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from btlight.data.panel import DynamicPanel
from btlight.ml.model.ensemble import (
    GaussianEnsemble,
    allocation_factor,
    marginal_quantile,
    mixture_moments,
    mixture_nll,
)
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
    """Model changing cross-sections with conditionally independent asset returns."""
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
            n, layer.risk_aversion, cost=layer.cost, diagonal=True
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
        # Each train/validation pass starts afresh; dates stay in chronological order.
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
                    # Match holdings by asset ID, including confirmed sales of departing assets.
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
                    # Retain a forecast-fitting term alongside the portfolio objective.
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

    # Use the best validation epoch rather than the final training epoch.
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
    cost=0.0,
    first_test="2014-01-01",
    smoke=False,
    prediction="point",
    members=1,
    finetune=True,
    initial_variance=1e-4,
    variance_floor=1e-8,
    correlation_shrinkage=0.1,
    nll_weight=1e-3,
    distribution_output=None,
    executed_exits=None,
    log=None,
):
    dynamic = isinstance(panel, DynamicPanel)
    if isinstance(members, bool) or not isinstance(members, int) or members < 1:
        raise ValueError("members must be a positive integer")
    if members != 1 and prediction != "gaussian":
        raise ValueError("multiple members require Gaussian prediction")
    if prediction not in {"point", "gaussian", "equal_weight"}:
        raise ValueError("prediction must be point, gaussian or equal_weight")
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("cost must be finite and nonnegative")
    equal_weight = prediction == "equal_weight"
    if not equal_weight and not pretrain and not finetune:
        raise ValueError("enable pretraining or E2E fine-tuning")
    strategy = "equal_weight" if equal_weight else ("e2e" if finetune else "pto")
    probabilistic = prediction == "gaussian" and not equal_weight
    pretrain = pretrain and not equal_weight
    if probabilistic:
        if not math.isfinite(nll_weight) or nll_weight <= 0:
            raise ValueError("nll_weight must be finite and positive")
    if distribution_output is not None:
        if not probabilistic or Path(distribution_output).suffix != ".npz":
            raise ValueError("distribution_output requires gaussian prediction and an .npz path")

    if log:
        log(
            "settings",
            prediction=None if equal_weight else prediction,
            finetune=finetune and not equal_weight,
            pretrain=pretrain,
            max_epochs=epochs,
            risk="none" if equal_weight else "variance",
            risk_aversion=0.0 if equal_weight else risk_aversion,
            cost=cost,
            first_test=first_test,
            initial_variance=initial_variance if probabilistic else None,
            variance_floor=variance_floor if probabilistic else None,
            nll_weight=(nll_weight if strategy == "e2e" else 0.0) if probabilistic else None,
            correlation_shrinkage=correlation_shrinkage if probabilistic and not dynamic else 0.0,
            covariance_model=(
                None
                if equal_weight
                else ("historical-correlation" if probabilistic and not dynamic
                      else "diagonal-plus-low-rank" if members > 1 else "diagonal")
            ),
            asset_cap=None if equal_weight else 0.1,
            seed=7,
            members=members,
            member_seeds=list(range(7, 7 + members)),
            dtype="float64",
            torch_threads=1,
            solver=None if equal_weight else "Clarabel",
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

    # Historical risk scores portfolios; Gaussian forecasts supply allocation risk.
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
        return PortfolioLayer(n, risk_aversion, cost=cost, diagonal=dynamic,
                              disagreement_rank=members if dynamic and members > 1 else 0)

    layer = (
        None
        if equal_weight
        else allocation_layer(int(panel.offsets[1]) if dynamic else len(panel.assets))
    )
    # Members train through their own allocations; test forecasts are combined.
    training_layer = (
        PortfolioLayer(int(panel.offsets[1]) if dynamic else len(panel.assets),
                       risk_aversion, cost=cost, diagonal=dynamic)
        if members > 1 else layer
    )
    records = []
    distribution_records = []
    forecast_scores = []
    previous = (
        torch.zeros(0, dtype=torch.float64)
        if dynamic
        else torch.full((len(panel.assets),), 1 / len(panel.assets), dtype=torch.float64)
    )
    # Test holdings persist across quarters even though each network is refitted.
    previous_ids, previous_end = [], None
    for quarter, train, val, test in experiment_splits(panel, first_test, factors, smoke):
        if dynamic:
            for ids in (train, val, test):
                panel.validate_portfolio_dates(ids)
        # Refit from initialization using only this window's train/validation dates.
        if equal_weight:
            model = None
        elif probabilistic:
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
        if strategy == "pto":
            schedule = f"{pretraining.upper()} training only, then allocation"
        elif equal_weight:
            schedule = "Equal weight; no training"
        print(
            f"{quarter}: {schedule} ({prediction})",
            flush=True,
        )
        fitted = []
        for member in range(members):
            seed = 7 + member
            if members > 1:
                torch.manual_seed(seed)
                model = GaussianReturnPredictor(
                    panel.features.shape[-1], initial_variance=initial_variance,
                    variance_floor=variance_floor,
                ).double()

            def member_log(event, **fields):
                if log:
                    log(event, member=member + 1, seed=seed, **fields)

            if pretrain:
                fit(
                    model,
                    panel,
                    train,
                    val,
                    factors,
                    training_layer,
                    epochs,
                    pretraining,
                    correlations=correlations,
                    nll_weight=nll_weight,
                    executed_exits=executed_exits,
                    log=member_log,
                    quarter=quarter,
                )
            if strategy == "e2e":
                fit(
                    model,
                    panel,
                    train,
                    val,
                    factors,
                    training_layer,
                    epochs,
                    "e2e",
                    correlations=correlations,
                    nll_weight=nll_weight,
                    executed_exits=executed_exits,
                    log=member_log,
                    quarter=quarter,
                )
            fitted.append(model)
        if members > 1:
            model = GaussianEnsemble(fitted).eval()

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
                    if not equal_weight:
                        allocator = allocation_layer(len(current_ids))

                disagreement = None
                components = None
                if equal_weight:
                    weights = torch.full_like(previous, 1 / len(features))
                elif probabilistic:
                    if members > 1:
                        components = model.predict_components(
                            features, None if dynamic else correlations[t]
                        )
                        mu, within, disagreement = mixture_moments(components)
                        allocation_risk = (
                            within.sqrt() if dynamic
                            else allocation_factor(components, disagreement)
                        )
                    else:
                        distribution = (
                            dynamic_distribution(model, features)
                            if dynamic else model.predict_distribution(features, correlations[t])
                        )
                        mu = distribution.mean
                        allocation_risk = distribution.stddev if dynamic else distribution.scale_tril.T
                    allocation_factor_value = allocation_risk
                else:
                    mu = model(features)
                    allocation_factor_value = (
                        torch.as_tensor(panel.risk_variance[sl], dtype=torch.float64).sqrt()
                        if dynamic
                        else factors[t]
                    )

                if not equal_weight:
                    weights = (
                        allocator(mu, allocation_factor_value, previous, disagreement)
                        if dynamic and members > 1
                        else allocator(mu, allocation_factor_value, previous)
                    )
                if probabilistic:
                    realized = torch.tensor(panel.targets[sl], dtype=torch.float64)
                    if components is not None:
                        nll = mixture_nll(components, realized).item()
                        lower = marginal_quantile(components, 0.025)
                        upper = marginal_quantile(components, 0.975)
                        coverage = ((realized >= lower) & (realized <= upper)).double().mean().item()
                        variance = within + disagreement.square().sum(0)
                    else:
                        nll = gaussian_nll(distribution, realized).item()
                        standardized = (realized - mu) / distribution.stddev
                        coverage = (standardized.abs() <= 1.959963984540054).double().mean().item()
                        variance = distribution.variance
                    forecast_scores.append((nll, coverage))
                    if distribution_output is not None:
                        distribution_records.append(
                            (
                                t,
                                mu.numpy(),
                                variance.numpy(),
                                allocation_factor_value.numpy(),
                                weights.numpy(),
                                np.stack([d.mean.numpy() for d in components]) if components else mu.numpy()[None],
                                np.stack([d.variance.numpy() for d in components]) if components else variance.numpy()[None],
                            )
                        )

                # Approximate turnover from weight changes, without return-drift adjustment.
                turnover = ((weights - previous).abs().sum() + exit_weight).item() / 2
                records.append((t, weights.numpy(), cost * turnover))
                previous = weights

        if log:
            log("window_end", quarter=quarter, test_dates=len(test))

    # Packed arrays preserve variable cross-section sizes in forecast exports.
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
            member_mean=np.concatenate([r[5] for r in distribution_records], axis=1),
            member_variance=np.concatenate([r[6] for r in distribution_records], axis=1),
            member_seeds=np.arange(7, 7 + members),
            mixture_weights=np.full(members, 1 / members),
            nll_per_asset=np.array(forecast_scores)[:, 0],
            mean=np.concatenate([r[1] for r in distribution_records]),
            variance=np.concatenate([r[2] for r in distribution_records]),
            within_variance=np.concatenate([r[6].mean(0) for r in distribution_records]),
            disagreement_factor=np.concatenate([
                (r[5] - r[1]) / math.sqrt(members) for r in distribution_records
            ], axis=1),
            weights=np.concatenate([r[4] for r in distribution_records]),
            targets=np.concatenate([panel.targets[panel.day(t)] for t in ids]),
            interpretation=f"dynamic Gaussian {strategy.upper()}; diagonal member covariance plus ensemble disagreement; confirmed exits only",
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
            member_mean=np.stack([r[5] for r in distribution_records]),
            member_variance=np.stack([r[6] for r in distribution_records]),
            member_seeds=np.arange(7, 7 + members),
            mixture_weights=np.full(members, 1 / members),
            mean=np.stack([r[1] for r in distribution_records]),
            variance=np.stack([r[2] for r in distribution_records]),
            within_variance=np.stack([r[6].mean(0) for r in distribution_records]),
            disagreement_factor=np.stack([
                (r[5] - r[1]) / math.sqrt(members) for r in distribution_records
            ]),
            covariance_factor=np.stack([r[3] for r in distribution_records]),
            weights=np.stack([r[4] for r in distribution_records]),
            targets=panel.targets[ids],
            nll_per_asset=np.array(forecast_scores)[:, 0],
            correlation_shrinkage=correlation_shrinkage,
            nll_weight=nll_weight if strategy == "e2e" else 0.0,
            initial_variance=initial_variance,
            variance_floor=variance_floor,
            interpretation="Gaussian mixture panel-target returns; no parameter posterior",
        )
        print(f"Saved return-distribution diagnostics to {path}")

    model_name = "Wang E2E" if finetune else "Point PTO"
    if probabilistic:
        model_name = f"Gaussian returns {strategy.upper()}"
        if members > 1:
            model_name += f" ensemble ({members})"
    if equal_weight:
        model_name = "Equal weight"
    result = summarize(panel, records).assign(
        model=model_name,
        strategy=strategy,
        risk="none" if equal_weight else "variance",
        risk_aversion=0.0 if equal_weight else risk_aversion,
        cost=cost,
        smoke=smoke,
        pretrain=pretrain,
    )
    if probabilistic:
        result = result.assign(
            prediction=prediction,
            members=members,
            nll_weight=nll_weight if strategy == "e2e" else 0.0,
            initial_variance=initial_variance,
            variance_floor=variance_floor,
            correlation_shrinkage=0.0 if dynamic else correlation_shrinkage,
            covariance_model=("diagonal-plus-low-rank" if members > 1 else "diagonal") if dynamic else "historical-correlation",
            test_nll_per_asset=np.mean(forecast_scores, axis=0)[0],
            marginal_coverage_95=np.mean(forecast_scores, axis=0)[1],
        )
    return result
