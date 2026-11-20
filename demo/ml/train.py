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
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from btlight.data.panel import DynamicPanel, load_data
from btlight.ml.training.e2e import run
from btlight.ml.training.portfolio_data import membership_exits, save_results


def git_commit():
    """Short commit hash of the code that produced a run, if available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def compact(value):
    # Three significant digits
    if isinstance(value, float):
        return float(f"{value:.3g}") if math.isfinite(value) else None
    return value


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
    parser.add_argument("--first-test", help="First test month; default 2006-01-01 on real data")
    parser.add_argument("--train-months", type=int, help="Rolling training window; default expanding")
    parser.add_argument("--validation-months", type=int, default=60)
    parser.add_argument("--test-months", type=int, default=12, help="Test months per retrain")
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
    windows_output = output.with_name(f"{name}_windows.csv")
    log_path.parent.mkdir(parents=True, exist_ok=False)
    commit = git_commit()
    context = {
        "model": args.model,
        "command": sys.argv,
        "git_commit": commit,
        "panel": str(Path(args.panel).resolve()) if args.panel else "synthetic-seed-7",
        "output": str(output),
        "windows_output": str(windows_output),
        "distribution_output": str(distribution_output) if distribution_output else None,
    }
    started = perf_counter()
    with log_path.open("x", buffering=1) as stream:

        def log(event, **fields):
            # run() logs its settings; the CLI adds where the run came from and goes to.
            if event == "run_start":
                fields = {**context, **fields}
            record = dict(
                timestamp=datetime.now(UTC).isoformat(),
                elapsed_seconds=perf_counter() - started,
                event=event,
                **fields,
            )
            record = {k: compact(v) for k, v in record.items()}
            stream.write(json.dumps(record, default=str, allow_nan=False) + "\n")
            stream.flush()

        print(f"Training log: {log_path}", flush=True)
        try:
            panel = load_data(args.panel, args.smoke)
            # The synthetic smoke panel is daily and short, so it keeps Wang's 39/9/3.
            synthetic = args.smoke and not args.panel
            windows = (
                {"train_months": 39, "validation_months": 9, "test_months": 3}
                if synthetic
                else {
                    "train_months": args.train_months,
                    "validation_months": args.validation_months,
                    "test_months": args.test_months,
                }
            )

            result = run(
                panel,
                pretrain=True,  # Set False to train E2E from random initialization.
                epochs=min(args.epochs, 2) if args.smoke else args.epochs,
                prediction=args.prediction,
                members=args.members,
                finetune=not args.no_e2e,
                risk_aversion=args.risk_aversion,
                cost=args.cost,
                first_test=args.first_test or ("2014-01-01" if synthetic else "2006-01-01"),
                **windows,
                smoke=args.smoke,
                distribution_output=distribution_output,
                executed_exits=membership_exits(panel) if isinstance(panel, DynamicPanel) else None,
                log=log,
            )
            # Save the performance summary; Gaussian forecasts are exported by run().
            windows = result.attrs.get("windows", [])
            result = result.assign(git_commit=commit, run_folder=output.parent.name)
            save_results([result], output, args.panel or "synthetic-seed-7")
            pd.DataFrame(windows).to_csv(windows_output, index=False)
        except BaseException as error:
            log(
                "run_end",
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error_type=type(error).__name__,
                message=str(error),
                seconds=perf_counter() - started,
            )
            raise
        log("run_end", status="completed", seconds=perf_counter() - started)


if __name__ == "__main__":
    main()
