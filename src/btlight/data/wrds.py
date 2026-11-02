"""Small US Compustat download adapted from data-wrds/download_sample.py."""

from configparser import ConfigParser
from pathlib import Path
from typing import Sequence

import pandas as pd
from sqlalchemy import URL, bindparam, create_engine, text
from sqlalchemy.engine import Engine

# Preserve the original USD conversion and annual-fundamentals issue filter.
SECURITY_QUERY = """
    WITH ccy_d AS (
        SELECT from_ccy.datadate, from_ccy.tocurd AS ccy,
               to_ccy.exratd / from_ccy.exratd AS exratd
        FROM comp.exrt_dly AS from_ccy
        JOIN comp.exrt_dly AS to_ccy
          ON from_ccy.datadate = to_ccy.datadate
         AND to_ccy.tocurd = 'USD'
    )
    SELECT s.gvkey, s.iid, s.datadate, c.conm, 'na' AS source,
           c.fic, s.tpci, s.exchg,
           s.prccd * cd.exratd AS prccd_ccy,
           s.prccd * cd.exratd * s.trfd / s.ajexdi AS prccd_ccy_adj,
           s.prccd * s.trfd / s.ajexdi AS prccd_adj,
           s.cshoc, s.cshtrd
    FROM comp.secd AS s
    JOIN ccy_d AS cd ON s.datadate = cd.datadate AND s.curcdd = cd.ccy
    JOIN comp.company AS c ON s.gvkey = c.gvkey
    WHERE EXISTS (
        SELECT 1 FROM comp.funda AS f
        WHERE s.gvkey = f.gvkey AND s.iid = f.iid
    )
      AND s.cshoc IS NOT NULL
      AND s.prccd IS NOT NULL
      AND s.fic = 'USA'
      AND s.gvkey IN :gvkeys
      AND s.datadate BETWEEN :history_start AND :end_date
    ORDER BY s.gvkey, s.iid, s.datadate
"""


def connect_to_db(config_file: Path) -> Engine:
    """Create an engine using a local INI file; connect only when a query is run."""
    config = ConfigParser(interpolation=None)
    if not config.read(config_file):
        raise FileNotFoundError(config_file)
    settings = config["wrds"]
    url = URL.create(
        "postgresql+psycopg2",
        username=settings["USERNAME"],
        password=settings["PASSWORD"],
        host=settings["HOST"],
        port=int(settings["PORT"]),
        database=settings["DATABASE"],
    )
    return create_engine(url)


def download_sample(
    engine: Engine,
    tickers: Sequence[str] = ("AAPL", "MSFT", "KO"),
    start: str = "2020-01-01",
    end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return security histories and matched companies for a small download check.

    Tickers select companies within the sample period, not a historical universe.
    Company/issue identifiers remain separate; no missing returns are filled.
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers or pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Supply tickers and an increasing date interval")
    params = {
        "tickers": tickers,
        "history_start": (pd.Timestamp(start) - pd.Timedelta(days=31)).date(),
        "end_date": pd.Timestamp(end).date(),
    }
    company_query = text("""
        SELECT DISTINCT c.gvkey, c.conm, s.tic
        FROM comp.secd AS s
        JOIN comp.company AS c ON c.gvkey = s.gvkey
        WHERE s.tic IN :tickers
          AND c.fic = 'USA'
          AND s.datadate BETWEEN :history_start AND :end_date
        ORDER BY s.tic
    """).bindparams(bindparam("tickers", expanding=True))
    companies = pd.read_sql_query(company_query, engine, params=params)
    if set(companies["tic"]) != set(tickers):
        raise ValueError("Not all requested tickers were found")
    if companies["tic"].duplicated().any():
        raise ValueError("Multiple companies matched a ticker; inspect the identifiers")

    params["gvkeys"] = companies["gvkey"].tolist()
    query = text(SECURITY_QUERY).bindparams(bindparam("gvkeys", expanding=True))
    data = pd.read_sql_query(query, engine, params=params, parse_dates=["datadate"])
    return prepare_security_data(data, start), companies


def prepare_security_data(data: pd.DataFrame, start: str) -> pd.DataFrame:
    """Calculate capitalization and total returns before trimming warm-up dates."""
    if data.empty:
        raise ValueError("The query returned no security data")
    data = data.sort_values(["gvkey", "iid", "datadate"]).copy()
    data["datadate"] = pd.to_datetime(data["datadate"])
    if data.duplicated(["gvkey", "iid", "datadate"]).any():
        raise ValueError("Duplicate security-date observations found")
    data["market_cap"] = data["prccd_ccy"] * data["cshoc"]
    data["return"] = data.groupby(["gvkey", "iid"])["prccd_ccy_adj"].pct_change(fill_method=None)
    data = data.loc[data["datadate"] >= pd.Timestamp(start)].copy()
    if data.empty:
        raise ValueError("No security observations in the requested interval")
    return data
