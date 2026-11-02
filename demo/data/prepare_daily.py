"""Prepare daily market/JKP inputs, preserving unknown returns and join audits."""

import argparse
import json
from pathlib import Path

import pandas as pd

from btlight.data.daily import (
    apply_stock_splits, check_stock_splits, prepare_daily, read_sources,
    trading_calendar, validate_stock_splits,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/market and jkp"))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output", type=Path, default=Path("data/daily.npz"))
    parser.add_argument("--split-reference", type=Path, help="Compustat security Parquet file/directory; saves candidate split audit")
    parser.add_argument("--split-fx", type=Path, help="Compustat FX rates for matching supplied EUR prices")
    parser.add_argument("--split-corrections", type=Path, help="Reviewed split audit; apply rows with decision=apply")
    parser.add_argument("--terminal-events", type=Path, help="Reviewed corporate-action proceeds in EUR, including explicit recovery assumptions")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace the prepared panel and audits"
    )
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("output must be an .npz file")
    if args.split_reference and args.split_corrections:
        parser.error("use split-reference for checking or split-corrections for applying")
    outputs = [
        args.output,
        *[
            args.output.with_suffix(s)
            for s in (
                ".audit.parquet",
                ".returns.parquet",
                ".exits.parquet",
                ".json",
                ".calendar.parquet",
            )
        ],
    ]
    split_output = args.output.with_suffix(".splits.parquet")
    if args.split_reference or args.split_corrections:
        outputs.append(split_output)
    if args.split_corrections and args.split_corrections.resolve() == split_output.resolve():
        parser.error("split-corrections must be separate from the output audit")
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("output or audit already exists; choose a new output filename")
    market, jkp = read_sources(args.source, args.start, args.end)
    splits = None
    if args.split_reference:
        security = pd.read_parquet(args.split_reference,
            columns=["gvkey", "iid", "datadate", "prccd", "ajexdi", "cshoc", "curcdd"],
            filters=[("gvkey", "in", market.company.unique().tolist())])
        scheduled, _ = trading_calendar(market)
        fx = pd.read_parquet(args.split_fx) if args.split_fx else None
        splits = check_stock_splits(market.loc[market.date.isin(scheduled)], security, fx)
        splits = validate_stock_splits(splits, security)
    if args.split_corrections:
        splits = pd.read_parquet(args.split_corrections)
        market = apply_stock_splits(market, splits)
    events = pd.read_parquet(args.terminal_events) if args.terminal_events else None
    panel, audit, review, exits, summary = prepare_daily(market, jkp, args.start, args.end, events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(outputs[0])
    audit.to_parquet(outputs[1], index=False)
    review.to_parquet(outputs[2], index=False)
    exits.to_parquet(outputs[3], index=False)
    summary.update(source=str(args.source), start=args.start, end=args.end)
    if args.terminal_events:
        summary.update(terminal_event_source=str(args.terminal_events),
                       terminal_policy="cash-equivalent proceeds at scheduled rebalance; no settlement delay; stopped orders remain cash; named distressed cases use explicit zero-recovery assumptions")
    if args.split_corrections:
        summary.update(
            split_corrections=str(args.split_corrections),
            applied_adjustments=int(splits.decision.eq("apply").sum()),
            return_basis="supplied-price returns with approved split/distribution adjustments; currency unverified",
            outliers="finite returns retained; absolute returns above 50% flagged for review only; signals rank-normalized",
        )
    outputs[4].write_text(json.dumps(summary, indent=2) + "\n")
    _, calendar = trading_calendar(market)
    calendar.to_parquet(outputs[5], index=False)
    if splits is not None:
        splits.to_parquet(split_output, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
