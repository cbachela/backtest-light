"""Run the Wang reproduction, optionally with a fixed-uncertainty BL update."""

import argparse
from pathlib import Path

from btlight.data.panel import load_data
from btlight.ml.training.portfolio_data import save_results
from btlight.ml.training.wang_e2e import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", help="Shared NPZ panel; omit only with --smoke")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--risk-aversion", type=float, nargs="+", default=[1.0])
    parser.add_argument("--output", help="Defaults to outputs/wang_e2e.csv or outputs/wang_bl.csv")
    parser.add_argument("--risk", choices=["sd", "variance"], default="sd")
    parser.add_argument("--cost", type=float, default=0.0)
    parser.add_argument("--first-test", default="2014-01-01")
    parser.add_argument(
        "--uncertainty-output",
        type=Path,
        help="Optional NPZ of rolling prediction-error estimates on test dates",
    )
    parser.add_argument(
        "--error-window",
        type=int,
        default=504,
        help="Observed residuals per estimate (default: about two trading years)",
    )
    parser.add_argument(
        "--bl",
        choices=["off", "constant", "historical"],
        default="off",
        help="Fixed Omega: scalar variance or training residual variances",
    )
    parser.add_argument(
        "--bl-variance",
        type=float,
        default=1e-4,
        help="Constant daily variance in decimal-return squared units",
    )
    parser.add_argument("--bl-tau", type=float, default=0.05)
    parser.add_argument(
        "--bl-factors", type=Path, help="Daily French factors CSV for return-basis conversion"
    )
    parser.add_argument(
        "--bl-market-weights",
        type=Path,
        help="CSV: date (availability date), then market weights for each panel asset",
    )
    parser.add_argument(
        "--bl-delta",
        type=float,
        default=2.5,
        help="Market risk aversion for the reverse CAPM prior",
    )
    args = parser.parse_args()
    if (
        args.bl != "off"
        and not args.smoke
        and (args.bl_factors is None or args.bl_market_weights is None)
    ):
        parser.error("BL on real data requires --bl-factors and --bl-market-weights")
    if args.uncertainty_output is not None:
        if args.uncertainty_output.suffix != ".npz":
            parser.error("uncertainty-output must end in .npz")
        if len(args.risk_aversion) != 1:
            parser.error("use one risk-aversion value when saving uncertainty")
        if args.error_window < 2:
            parser.error("error-window must be at least 2")
    panel = load_data(args.panel, args.smoke)
    results = [
        run(
            panel,
            epochs=min(args.epochs, 2) if args.smoke else args.epochs,
            risk_aversion=eta,
            risk=args.risk,
            cost=args.cost,
            first_test=args.first_test,
            smoke=args.smoke,
            uncertainty_output=args.uncertainty_output,
            error_window=args.error_window,
            bl_mode=args.bl,
            bl_variance=args.bl_variance,
            bl_tau=args.bl_tau,
            bl_factors=args.bl_factors,
            bl_market_weights=args.bl_market_weights,
            bl_delta=args.bl_delta,
        )
        for eta in args.risk_aversion
    ]
    output = args.output or ("outputs/wang_e2e.csv" if args.bl == "off" else "outputs/wang_bl.csv")
    save_results(results, output, args.panel or "synthetic-seed-7")


if __name__ == "__main__":
    main()
