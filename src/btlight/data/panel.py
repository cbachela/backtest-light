"""Fixed or changing-universe panels with explicit feature/label availability."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass
class Panel:
    dates: np.ndarray  # signal-availability dates
    label_end: np.ndarray  # dates when realized targets become observable
    features: np.ndarray  # [dates, assets, features], already point-in-time
    targets: np.ndarray  # forward benchmark-relative returns
    risk_returns: np.ndarray  # trailing benchmark-relative returns known on dates
    raw_returns: np.ndarray  # forward raw returns for wealth/drawdown
    assets: np.ndarray

    def __post_init__(self):
        self.dates = np.asarray(self.dates, dtype="datetime64[D]")
        self.label_end = np.asarray(self.label_end, dtype="datetime64[D]")
        t, n, _ = self.features.shape
        if n < 2 or len(set(self.assets)) != n:
            raise ValueError("at least two distinct assets are required")
        if self.dates.shape != (t,) or self.label_end.shape != (t,):
            raise ValueError("date shape mismatch")
        if np.isnat(self.dates).any() or np.isnat(self.label_end).any():
            raise ValueError("missing dates")
        if not np.all(np.diff(self.dates) > np.timedelta64(0, "D")):
            raise ValueError("signal dates must be strictly increasing")
        if not np.all(self.label_end > self.dates):
            raise ValueError("labels must become observable after signal dates")
        for a in (self.targets, self.risk_returns, self.raw_returns):
            if a.shape != (t, n):
                raise ValueError("return shape mismatch")
        if not all(
            np.isfinite(a).all()
            for a in (self.features, self.targets, self.risk_returns, self.raw_returns)
        ):
            raise ValueError("missing/nonfinite data: supply an explicit preprocessing policy")
        if (self.raw_returns <= -1).any():
            raise ValueError("raw simple returns must exceed -1")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if "offsets" in data:
                return DynamicPanel(**{key: data[key] for key in DynamicPanel.__dataclass_fields__})
            return cls(**{key: data[key] for key in cls.__dataclass_fields__})

    def save(self, path):
        np.savez_compressed(path, **vars(self))


MARKET_INPUTS = (
    "return_1",
    "return_5",
    "return_21",
    "momentum_63_21",
    "volatility_21",
    "volatility_63",
    "max_return_21",
    "turnover_63",
)
ACCOUNTING_INPUTS = (
    "income_assets",
    "operating_income_assets",
    "cashflow_assets",
    "cash_assets",
    "debt_assets",
    "sales_assets",
    "asset_growth",
    "liabilities_assets",
)


@dataclass
class DynamicPanel:
    """Packed date cross-sections: absence means ineligible, never a zero return.

    Features contain only contemporaneous information. Missing labels remain in
    the panel; training masks them separately. No rectangular survivor selection.
    """

    dates: np.ndarray
    offsets: np.ndarray
    assets: np.ndarray
    asset_index: np.ndarray
    features: np.ndarray
    feature_names: np.ndarray
    targets: np.ndarray
    entry_dates: np.ndarray
    label_end: np.ndarray
    risk_variance: np.ndarray

    def __post_init__(self):
        self.dates = np.asarray(self.dates, dtype="datetime64[D]")
        self.entry_dates = np.asarray(self.entry_dates, dtype="datetime64[D]")
        self.label_end = np.asarray(self.label_end, dtype="datetime64[D]")
        rows = len(self.targets)
        if (
            not len(self.dates)
            or np.isnat(self.dates).any()
            or not np.all(np.diff(self.dates) > np.timedelta64(0, "D"))
        ):
            raise ValueError("nonempty, increasing signal dates required")
        if (
            self.offsets.shape != (len(self.dates) + 1,)
            or self.offsets[0] != 0
            or self.offsets[-1] != rows
            or (np.diff(self.offsets) < 2).any()
        ):
            raise ValueError("invalid daily offsets; need at least two eligible companies per date")
        if (
            self.features.shape != (rows, len(self.feature_names))
            or not np.isfinite(self.features).all()
        ):
            raise ValueError("finite features and matching feature names required")
        if len(set(self.assets)) != len(self.assets) or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("duplicate assets or feature names")
        for a in (
            self.asset_index,
            self.entry_dates,
            self.label_end,
            self.risk_variance,
        ):
            if a.shape != (rows,):
                raise ValueError("row shape mismatch")
        if not np.issubdtype(self.offsets.dtype, np.integer) or not np.issubdtype(
            self.asset_index.dtype, np.integer
        ):
            raise ValueError("offsets and asset indices must be integers")
        if (self.asset_index < 0).any() or (self.asset_index >= len(self.assets)).any():
            raise ValueError("unknown asset index")
        if not np.isfinite(self.risk_variance).all() or (self.risk_variance <= 0).any():
            raise ValueError("positive known risk variances required")
        if np.isinf(self.targets).any() or (self.targets[np.isfinite(self.targets)] < -1).any():
            raise ValueError("invalid raw return targets")
        for t, date in enumerate(self.dates):
            sl = self.day(t)
            if len(np.unique(self.asset_index[sl])) != sl.stop - sl.start:
                raise ValueError("duplicate asset/date")
            known = np.isfinite(self.targets[sl])
            if (known & (np.isnat(self.entry_dates[sl]) | np.isnat(self.label_end[sl]))).any():
                raise ValueError("observed targets need entry and end dates")
            if not (
                (self.entry_dates[sl][known] > date)
                & (self.label_end[sl][known] > self.entry_dates[sl][known])
            ).all():
                raise ValueError("target chronology mismatch")

    def day(self, t):
        return slice(int(self.offsets[t]), int(self.offsets[t + 1]))

    def label_mask(self, t, before=None):
        sl = self.day(t)
        mask = np.isfinite(self.targets[sl]) & ~np.isnat(self.label_end[sl])
        if before is not None:
            mask &= self.label_end[sl] < np.datetime64(before, "D")
        return mask

    def validate_portfolio_dates(self, indices):
        """Check outcomes without selecting assets using their future availability."""
        last_end = None
        for t in indices:
            sl = self.day(t)
            if not self.label_mask(t).all():
                raise ValueError(
                    f"Missing portfolio outcomes on {self.dates[t]}; execution cannot be valued"
                )
            if len(np.unique(self.entry_dates[sl])) != 1 or len(np.unique(self.label_end[sl])) != 1:
                raise ValueError(f"Unaligned execution dates on {self.dates[t]}")
            entry, end = self.entry_dates[sl][0], self.label_end[sl][0]
            if last_end is not None and entry != last_end:
                raise ValueError(f"Nonconsecutive execution intervals on {self.dates[t]}")
            last_end = end

    def save(self, path):
        np.savez_compressed(path, **vars(self))

    @classmethod
    def from_cleaned(cls, path):
        """Use screened data, existing sector/date fills, then neutral placeholders.

        Accounting zero placeholders are accompanied by original missing flags;
        they are model inputs, not repairs to the underlying financial statements.
        Market history is required. Eligibility never depends on future labels.
        """
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        if (
            manifest.get("currency") != "EUR"
            or manifest.get("size_screen", {}).get("fraction") != 0.2
        ):
            raise ValueError("expected EUR dataset with the approved 20% size screen")
        signals = MARKET_INPUTS + ACCOUNTING_INPUTS
        scores = [f + "_normal" for f in signals]
        flags = [f + "_missing" for f in ACCOUNTING_INPUTS]
        columns = [
            "gvkey",
            "iid",
            "signal_date",
            "entry_date",
            "label_end",
            "target_return_eur",
            "volatility_63",
            *scores,
            *flags,
        ]
        parts = []
        source_rows = 0
        for file in sorted((path / "daily").glob("*.parquet")):
            data = pd.read_parquet(file, columns=columns)
            source_rows += len(data)
            valid = np.isfinite(data[[f + "_normal" for f in MARKET_INPUTS]]).all(axis=1)
            valid &= np.isfinite(data.volatility_63) & data.volatility_63.ge(0)
            data = data.loc[valid].copy()
            for f in ACCOUNTING_INPUTS:
                missing = ~np.isfinite(data[f + "_normal"])
                data[f + "_missing"] = data[f + "_missing"].astype(bool) | missing
                data.loc[missing, f + "_normal"] = 0.0
            parts.append(data)
            print(f"{file.stem}: {len(data):,} eligible rows", flush=True)
        if not parts:
            raise ValueError("no daily partitions")
        data = pd.concat(parts, ignore_index=True).sort_values(["signal_date", "gvkey", "iid"])
        # Small cross-sections cannot support the shared network's batch statistics.
        data = data[data.groupby("signal_date").gvkey.transform("size").ge(10)]
        if data.empty or data.duplicated(["signal_date", "gvkey"]).any():
            raise ValueError("empty sample or duplicate company/date")
        ids, assets = pd.factorize(data.gvkey.astype(str) + ":" + data.iid.astype(str), sort=True)
        counts = data.groupby("signal_date", sort=True).size()
        print(
            f"Retained {len(data):,}/{source_rows:,} rows; {len(assets):,} securities; {len(counts):,} dates",
            flush=True,
        )
        return cls(
            dates=counts.index.to_numpy(dtype="datetime64[D]"),
            offsets=np.r_[0, counts.to_numpy().cumsum()],
            assets=assets.to_numpy(dtype=str),
            asset_index=ids.astype(np.int32),
            features=data[scores + flags].to_numpy(dtype=np.float32),
            feature_names=np.array(scores + flags),
            targets=data.target_return_eur.to_numpy(dtype=np.float64),
            entry_dates=data.entry_date.to_numpy(dtype="datetime64[D]"),
            label_end=data.label_end.to_numpy(dtype="datetime64[D]"),
            risk_variance=np.maximum(data.volatility_63.to_numpy() ** 2 / 252, 1e-8),
        )


def synthetic_panel(seed=7, periods=1400, assets=20, features=6):
    """Synthetic diagnostics only: targets end two business dates after the signal."""
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range("2010-01-01", periods=periods + 2).values.astype("datetime64[D]")
    x = rng.normal(size=(periods, assets, features))
    target = 0.003 * x[:, :, 0] + rng.normal(0, 0.015, (periods, assets))
    market = rng.normal(0.0002, 0.008, (periods, 1))
    # At date t, the label generated at t-2 has just become known.
    known = np.concatenate([rng.normal(0, 0.015, (2, assets)), target[:-2]])
    return Panel(
        calendar[:-2],
        calendar[2:],
        x,
        target,
        known,
        target + market,
        np.array([f"SYN{i:03}" for i in range(assets)]),
    )


def load_data(path=None, smoke=False):
    if smoke and path is None:
        return synthetic_panel()
    if path is not None:
        if Path(path).is_dir():
            return DynamicPanel.from_cleaned(path)
        return Panel.load(path)
    raise ValueError("choose --panel PATH or --smoke")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare changing-universe model inputs")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new filename")
    panel = DynamicPanel.from_cleaned(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(args.output)
    print(f"Saved {args.output}")
