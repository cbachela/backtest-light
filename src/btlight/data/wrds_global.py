"""Download raw Compustat Global data; investment-universe filters come later."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import bindparam, inspect, text

EUROPE = "AUT BEL BGR HRV CYP CZE DNK EST FIN FRA DEU GRC HUN IRL ITA LVA LTU LUX MLT NLD POL PRT ROU SVK SVN ESP SWE GBR CHE NOR ISL".split()
DAILY_FIELDS = "gvkey iid datadate prccd ajexdi trfd cshoc curcdd cshtrd prcstd prchd prcld qunit epf tpci exchg fic loc secstat isin sedol gind gsubind".split()
NUMERIC_FIELDS = set("prccd ajexdi trfd cshoc cshtrd prchd prcld".split())
REQUIRED_FIELDS = set("gvkey iid datadate prccd ajexdi trfd cshoc curcdd".split())
REFERENCE_TABLES = ("g_company", "g_security", "g_sec_history", "r_ex_codes", "r_country")


def identifier(name):
    """Only allow simple SQL identifiers; data values use bound parameters."""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return f'"{name}"'


def columns_for(engine, schema, table):
    inspector = inspect(engine)
    if not inspector.has_table(table, schema=schema):
        return []
    return [column["name"] for column in inspector.get_columns(table, schema=schema)]


def select_fields(columns, numeric=()):
    fields = []
    for name in columns:
        kind = "DOUBLE PRECISION" if name in numeric else "TEXT"
        fields.append(f"CAST(s.{identifier(name)} AS {kind}) AS {identifier(name)}")
    return ", ".join(fields)


def arrow_schema(columns, numeric=()):
    return pa.schema([(name, pa.float64() if name in numeric else pa.string()) for name in columns])


def geographic_filter(schema, daily, company, security, all_global, exchanges):
    if all_global:
        return "TRUE"
    prefix = identifier(schema)
    clauses = []
    for name in ("fic", "loc"):
        if name in daily:
            clauses.append(f"s.{name} IN :countries")
    country_columns = [name for name in ("fic", "loc") if name in company]
    if "gvkey" in company and country_columns:
        conditions = " OR ".join(f"c.{name} IN :countries" for name in country_columns)
        clauses.append(
            f"EXISTS (SELECT 1 FROM {prefix}.g_company c "
            f"WHERE c.gvkey = s.gvkey AND ({conditions}))"
        )
    if {"gvkey", "iid", "excntry"}.issubset(security):
        clauses.append(
            f"EXISTS (SELECT 1 FROM {prefix}.g_security c "
            "WHERE c.gvkey = s.gvkey AND c.iid = s.iid "
            "AND c.excntry IN :countries)"
        )
    elif not exchanges:
        raise ValueError(
            "No security exchange-country field found. Use --inspect to examine "
            "the schema, then provide European --exchange-codes or use --all-global."
        )
    if exchanges:
        if "exchg" not in daily:
            raise ValueError("The daily table has no exchg column")
        clauses.append("CAST(s.exchg AS TEXT) IN :exchanges")
    return "(" + " OR ".join(clauses) + ")"


def save_query(engine, query, params, path, schema, refresh=False):
    """Stream a query to a temporary file, exposing only completed Parquet files."""
    if path.exists() and not refresh:
        print(f"Already downloaded: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".parquet.partial")
    rows = 0
    with engine.connect().execution_options(stream_results=True) as connection:
        with pq.ParquetWriter(partial, schema, compression="zstd") as writer:
            for chunk in pd.read_sql_query(query, connection, params=params, chunksize=100_000):
                table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
                writer.write_table(table)
                rows += len(chunk)
    partial.replace(path)
    print(f"Saved {path}: {rows:,} rows", flush=True)


def download_global(
    engine,
    output,
    start,
    end,
    schema="comp",
    all_global=False,
    exchanges=(),
    refresh=False,
    extra_references=(),
):
    output = Path(output)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Supply a valid increasing date interval")
    prefix = identifier(schema)
    daily = columns_for(engine, schema, "g_secd")
    missing = REQUIRED_FIELDS.difference(daily)
    if missing:
        raise ValueError(f"Missing required g_secd columns: {sorted(missing)}")
    company = columns_for(engine, schema, "g_company")
    security = columns_for(engine, schema, "g_security")
    where = geographic_filter(schema, daily, company, security, all_global, exchanges)
    columns = [name for name in DAILY_FIELDS if name in daily]
    print(f"Unavailable optional daily fields: {sorted(set(DAILY_FIELDS) - set(columns))}")
    fx_columns = columns_for(engine, schema, "exrt_dly")
    if not {"datadate", "fromcurd", "tocurd", "exratd"}.issubset(fx_columns):
        raise ValueError(
            "Expected exrt_dly fields are missing; inspect the schema before downloading"
        )

    reference_names = list(dict.fromkeys([*REFERENCE_TABLES, *extra_references]))
    references = {}
    for table in reference_names:
        identifier(table)
        fields = columns_for(engine, schema, table)
        if fields:
            references[table] = fields
        elif table in extra_references:
            raise ValueError(f"Requested reference table is unavailable: {table}")
        else:
            print(f"Reference table unavailable, skipped: {table}")

    # Avoid silently combining downloads made with different filters or date bounds.
    settings = dict(
        schema=schema,
        start=str(start.date()),
        end=str(end.date()),
        all_global=all_global,
        countries=EUROPE,
        exchanges=sorted(map(str, exchanges)),
        daily_columns=columns,
        references=references,
        fx_columns=fx_columns,
        format_version=2,
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "download.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["settings"] != settings:
            raise ValueError("Download settings differ. Choose a new --output-dir.")
    else:
        manifest.write_text(
            json.dumps(
                {"settings": settings, "started_utc": datetime.now(timezone.utc).isoformat()},
                indent=2,
            )
        )

    # Keep reference snapshots separate: current headers are not historical classifications.
    for table, fields in references.items():
        query = text(f"SELECT {select_fields(fields)} FROM {prefix}.{identifier(table)} s")
        save_query(
            engine,
            query,
            {},
            output / "reference" / f"{table}.parquet",
            arrow_schema(fields),
            refresh,
        )

    statement = (
        f"SELECT {select_fields(columns, NUMERIC_FIELDS)} FROM {prefix}.g_secd s "
        f"WHERE s.datadate BETWEEN :start AND :end AND {where}"
    )
    query = text(statement)
    params = {}
    if ":countries" in statement:
        query = query.bindparams(bindparam("countries", expanding=True))
        params["countries"] = EUROPE
    if ":exchanges" in statement:
        query = query.bindparams(bindparam("exchanges", expanding=True))
        params["exchanges"] = list(map(str, exchanges))
    fx_query = text(
        f"SELECT {select_fields(fx_columns)} FROM {prefix}.exrt_dly s "
        "WHERE s.datadate BETWEEN :start AND :end"
    )
    for year in range(start.year, end.year + 1):
        dates = {
            "start": max(start, pd.Timestamp(year, 1, 1)).date(),
            "end": min(end, pd.Timestamp(year, 12, 31)).date(),
        }
        save_query(
            engine,
            query,
            {**params, **dates},
            output / "security" / f"{year}.parquet",
            arrow_schema(columns, NUMERIC_FIELDS),
            refresh,
        )
        save_query(
            engine,
            fx_query,
            dates,
            output / "fx" / f"{year}.parquet",
            arrow_schema(fx_columns),
            refresh,
        )
