################################
# Logging
################################
import logging
from rich.console import Console
from rich.logging import RichHandler

# simple logging for this experiment script
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True)],
)
logger = logging.getLogger(__name__)


################################
# Data paths
################################
from pathlib import Path
import pandas as pd

# basic paths
repo_root = Path(__file__).resolve().parents[2]
data_path = repo_root / "data" / "Compustat" / "switzerland"

# Load market and jkp data from parquet files
market_data = pd.read_parquet(path=data_path / "market_data.parquet")
jkp_data = pd.read_parquet(path=data_path / "jkp_data.parquet")

logger.info("market data and jkp data loaded")

################################
# Generate FFill for Market Data
################################

# re-using Cyril's function to align and forward fill
market_data_dates = market_data.index.get_level_values("date").unique().sort_values()
jkp_data_dates = jkp_data.index.get_level_values("date").unique().sort_values()
missing_dates = jkp_data_dates[~jkp_data_dates.isin(market_data_dates)]
tmp_dict = {}
for date in missing_dates:
    last_date = market_data_dates[market_data_dates <= date][-1]
    tmp_dict[date] = market_data.loc[last_date]

df_missing = pd.concat(tmp_dict, axis=0)
df_missing.index.names = market_data.index.names
market_data_ffill = pd.concat([market_data, df_missing]).sort_index()
market_data_ffill.index.names = ["DATE", "ID"]

# TODO: we have market_data with na as ID??

# daily returns
X = market_data_ffill.pivot_table(index="DATE", columns="ID", values="price")
end_date = X.index.max().strftime("%Y-%m-%d")
width = X.shape[0] - 1
daily_ret = X[X.index <= end_date].tail(width + 1).pct_change(fill_method=None).iloc[1:]
daily_ret = daily_ret.stack()
daily_ret.index.names = ["DATE", "ID"]
daily_ret.name = "tot_return_gross"
daily_ret.dropna().to_frame().to_parquet(data_path / "return_series.parquet")

logger.info("daily returns saved to return_series.parquet")
logger.info("script done")