"""Download broad European candidates from Compustat Global, without investment filters."""

import argparse
from pathlib import Path

from sqlalchemy import inspect

from btlight.data.wrds import connect_to_db
from btlight.data.wrds_global import columns_for, download_global


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--schema", default="comp")
    parser.add_argument("--output-dir", type=Path, default=Path("data/compustat_europe"))
    parser.add_argument(
        "--inspect", action="store_true", help="List available columns without downloading"
    )
    parser.add_argument(
        "--all-global", action="store_true", help="Keep all countries; a much larger download"
    )
    parser.add_argument(
        "--exchange-codes", nargs="*", default=[], help="Additional European exchange codes"
    )
    parser.add_argument(
        "--reference-tables",
        nargs="*",
        default=[],
        help="Additional history/reference tables to save in full",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace completed files with the same download settings",
    )
    args = parser.parse_args()
    engine = connect_to_db(args.config)
    try:
        if args.inspect:
            inspector = inspect(engine)
            tables = set(inspector.get_table_names(schema=args.schema))
            tables.update(inspector.get_view_names(schema=args.schema))
            names = {
                "g_secd",
                "g_company",
                "g_security",
                "g_sec_history",
                "exrt_dly",
                "r_ex_codes",
                "r_country",
            }
            names.update(name for name in tables if name.startswith("g_") and "hist" in name)
            for table in sorted(names):
                print(
                    f"{args.schema}.{table}: {', '.join(columns_for(engine, args.schema, table)) or 'unavailable'}"
                )
            return
        download_global(
            engine,
            args.output_dir,
            args.start,
            args.end,
            args.schema,
            args.all_global,
            args.exchange_codes,
            args.refresh,
            args.reference_tables,
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
