"""Prepare the monthly JKP panel: month-end signals and next-month returns."""

import argparse
import json
from pathlib import Path

from btlight.data.jkp import prepare_jkp, read_jkp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/market and jkp/df_jkp.parquet"))
    parser.add_argument("--start", default="1990-12-31", help="First month-end (first JKP eom)")
    parser.add_argument("--end", default="2025-12-31", help="Last month-end (last JKP eom)")
    parser.add_argument("--output", type=Path, default=Path("data/monthly.npz"))
    parser.add_argument("--size-fraction", type=float, default=0.2,
                        help="Smallest share of companies by market equity dropped each month")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("output must be an .npz file")
    audit_path, summary_path = args.output.with_suffix(".audit.parquet"), args.output.with_suffix(".json")
    if not args.overwrite and any(p.exists() for p in (args.output, audit_path, summary_path)):
        parser.error("output exists; use --overwrite or choose a new filename")

    panel, audit, summary = prepare_jkp(read_jkp(args.source, args.start, args.end), args.size_fraction)
    summary.update(source_file=str(args.source), start=args.start, end=args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(args.output)
    audit.to_parquet(audit_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
