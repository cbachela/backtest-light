"""Compare finished runs: one metrics table plus a CAGR bar chart and wealth paths.

Pass run folders, or none to use the newest finished run of each model in outputs/.
All runs must cover the same test months. Writes outputs/comparison_<timestamp>/.
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from btlight.ml.training.portfolio_data import performance
from plot_results import GRID, INK, log_axis, read_returns, style

# Fixed model order, so a model keeps its color across comparisons.
ORDER = [
    "baseline_pto", "baseline_e2e", "gaussian_pto", "gaussian_e2e",
    "gaussian_pto_ensemble5", "gaussian_e2e_ensemble5",
]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
BENCHMARK = "equal_weight"


def run_name(folder):
    # Folder names are <date>_<time>_<utc offset>_<name>.
    parts = folder.name.split("_", 3)
    return parts[3] if len(parts) == 4 else None


def latest_runs(root):
    runs = {}
    for folder in sorted(root.iterdir()):  # timestamps sort chronologically; newest wins
        name = run_name(folder) if folder.is_dir() else None
        if name and (folder / f"{name}_returns.csv").is_file():
            runs[name] = folder
    return list(runs.values())


def compare(folders, output):
    rows, paths, dates = [], {}, None
    for folder in folders:
        name = run_name(folder)
        returns = read_returns(folder / f"{name}_returns.csv")
        if dates is None:
            dates, benchmark = returns.date, returns
        elif not returns.date.equals(dates):
            raise ValueError(f"{folder.name} covers different test months")
        summary = pd.read_csv(folder / f"{name}.csv").iloc[0].to_dict()
        rows.append({"run": name, **summary})
        paths[name] = (1 + returns.portfolio).cumprod()
    # The equal-weight benchmark is in every returns file; add it unless that run is present.
    if BENCHMARK not in paths:
        gap = np.median(np.diff(benchmark.realized).astype("timedelta64[D]").astype(int))
        stats = performance(benchmark.equal_weight, 12 if gap >= 20 else 252)
        rows.append({"run": BENCHMARK, "model": "Equal-weight benchmark",
                     "observations": len(benchmark), **stats})
        paths[BENCHMARK] = (1 + benchmark.equal_weight).cumprod()
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "comparison.csv", index=False)

    models = sorted((r for r in paths if r != BENCHMARK),
                    key=lambda r: ORDER.index(r) if r in ORDER else len(ORDER))
    if len(models) > len(PALETTE):
        raise ValueError("more models than chart colors; compare fewer runs")
    colors = {**dict(zip(models, PALETTE)), BENCHMARK: INK}
    labels = dict(zip(table.run, table.model))
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), width_ratios=[1, 1.4])

    ranked = table.sort_values("cagr")
    left.barh(ranked.model, ranked.cagr, height=0.6,
              color=[INK if r == BENCHMARK else PALETTE[0] for r in ranked.run])
    for y, value in enumerate(ranked.cagr):
        left.annotate(f"{value:.1%}", (value, y), xytext=(4 if value >= 0 else -4, 0),
                      textcoords="offset points", va="center",
                      ha="left" if value >= 0 else "right", fontsize=8, color=INK)
    left.margins(x=0.2)  # room for the value labels at both ends
    left.axvline(0, color=INK, linewidth=0.8)
    left.xaxis.set_major_formatter(PercentFormatter(1))
    style(left, "Compound annual growth rate (CAGR)")
    left.grid(axis="y", visible=False)
    left.grid(axis="x", color=GRID, linewidth=0.8)

    for run in models + [BENCHMARK]:
        right.plot(benchmark.realized, paths[run], color=colors[run], linewidth=2,
                   linestyle="--" if run == BENCHMARK else "-", label=labels[run])
    log_axis(right)
    right.legend(frameon=False, fontsize=8)
    style(right, "Growth of 1 (log scale)")

    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    folders = args.folders or latest_runs(args.root)
    if not folders:
        raise SystemExit(f"no finished runs with a returns file under {args.root}")
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Saved {compare(folders, args.root / f'comparison_{stamp}')}")
