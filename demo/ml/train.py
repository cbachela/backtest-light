"""Run walk-forward portfolio experiments and save results with a training log.

--model gaussian: predicts return means and variances; baseline predicts means.
--model equal_weight: provides a comparison without training or optimization.
--no-e2e: trains forecasts only (PTO) for either learned model; 
--cost sets the turnover cost coefficient. 
Optimized portfolios use variance risk.

The shared training loop is implemented in e2e.py.

Set pretrain=True/Fase below to toggle MSE (point) or NLL (Gaussian) pretraining before E2E.
PTO requires pretraining, equal_weight ignores both training stages.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from btlight.data.panel import DynamicPanel, load_data
from btlight.ml.training.e2e import run
from btlight.ml.training.portfolio_data import exit_lookup, save_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", help="Prepared NPZ or cleaned European data directory")
    parser.add_argument("--smoke", action="store_true", help="Short wiring check")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--model",
        choices=["baseline", "gaussian", "equal_weight"],
        default="gaussian",
    )
    parser.add_argument("--members", type=int, default=1, help="Gaussian ensemble size (default 1)")
    parser.add_argument("--no-e2e", action="store_true", help="Train forecasts only, then allocate")
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--cost", type=float, default=0.0)
    parser.add_argument("--first-test", help="First test quarter; default 2024-01-01 on real data")
    parser.add_argument(
        "--exit-audit", type=Path, help="Audited eligibility exits with sale prices and dates"
    )
    args = parser.parse_args()

    # Translate the CLI baseline name to the shared trainer's point model.
    args.prediction = "point" if args.model == "baseline" else args.model
    name = (
        args.model
        if args.model == "equal_weight"
        else f"{args.model}_{'pto' if args.no_e2e else 'e2e'}"
    )
    if args.members > 1:
        name += f"_ensemble{args.members}"
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f_%z")
    output = Path("outputs") / f"{stamp}_{name}" / f"{name}.csv"
    distribution_output = output.with_suffix(".npz") if args.prediction == "gaussian" else None

    # Model constants live in Python; the CLI controls the experiment only.
    if args.members < 1 or (args.members != 1 and args.model != "gaussian"):
        parser.error("members must be positive; multiple members require --model gaussian")
    if args.epochs < 1:
        parser.error("epochs must be positive")

    # Use a fresh log for each run so partial progress survives interruption.
    log_path = output.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    with log_path.open("x", buffering=1) as stream:

        def log(event, **fields):
            record = dict(
                timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=perf_counter() - started,
                event=event,
                **fields,
            )
            stream.write(json.dumps(record, default=str, allow_nan=False) + "\n")
            stream.flush()

        print(f"Training log: {log_path}", flush=True)
        log(
            "experiment_start",
            model=args.model,
            command=sys.argv,
            panel=str(Path(args.panel).resolve()) if args.panel else "synthetic-seed-7",
            exit_audit=str(args.exit_audit) if args.exit_audit else None,
            output=str(output),
            distribution_output=str(distribution_output) if distribution_output else None,
        )
        try:
            # Load the panel and any confirmed exits before starting walk-forward training.
            panel = load_data(args.panel, args.smoke)
            if args.exit_audit is not None and not isinstance(panel, DynamicPanel):
                parser.error("exit-audit requires a changing-universe panel")

            result = run(
                panel,
                pretrain=True,  # Set False to train E2E from random initialization.
                epochs=min(args.epochs, 2) if args.smoke else args.epochs,
                prediction=args.prediction,
                members=args.members,
                finetune=not args.no_e2e,
                risk_aversion=args.risk_aversion,
                cost=args.cost,
                first_test=args.first_test
                or ("2014-01-01" if args.smoke and not args.panel else "2024-01-01"),
                smoke=args.smoke,
                distribution_output=distribution_output,
                executed_exits=exit_lookup(panel, args.exit_audit) if args.exit_audit else None,
                log=log,
            )
            # Save the performance summary; Gaussian forecasts are exported by run().
            save_results([result], output, args.panel or "synthetic-seed-7")
        except BaseException as error:
            log(
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error_type=type(error).__name__,
                message=str(error),
            )
            raise
        log("experiment_end", output=str(output))


if __name__ == "__main__":
    main()
