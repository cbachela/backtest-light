"""Plot one run's test returns against the equal-weight benchmark.

Four panels from <name>_returns.csv: growth of 1, cumulative performance relative to
the benchmark, drawdown and calendar-year returns. JKP returns are excess returns.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Categorical slots 1-2 of the default chart palette; text and grid stay neutral.
PORTFOLIO, BENCHMARK = "#2a78d6", "#eb6834"
INK, GRID = "#52514e", "#e4e3df"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(which="both", colors=INK, labelsize=8)


def log_axis(ax):
    # Plain numbers (0.8, 1, 2) rather than powers of ten on a log scale.
    ax.set_yscale("log")
    for axis in (ax.yaxis.set_major_formatter, ax.yaxis.set_minor_formatter):
        axis(FuncFormatter(lambda value, _: f"{value:g}"))


def read_returns(path):
    returns = pd.read_csv(path, parse_dates=["label_end"])
    # A return is realized over the period ending the day before its label end.
    returns["realized"] = returns.label_end - pd.Timedelta(days=1)
    return returns


def plot_run(path, output):
    returns = read_returns(path)
    series = {"Portfolio": ("portfolio", PORTFOLIO), "Equal weight": ("equal_weight", BENCHMARK)}
    wealth = (1 + returns[["portfolio", "equal_weight"]]).cumprod()
    when = returns.realized
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    for label, (column, color) in series.items():
        ax.plot(when, wealth[column], color=color, linewidth=2, label=label)
    log_axis(ax)
    style(ax, "Growth of 1 (log scale)")

    ax = axes[0, 1]
    ax.plot(when, wealth.portfolio / wealth.equal_weight - 1, color=PORTFOLIO, linewidth=2)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    style(ax, "Cumulative performance relative to equal weight")

    ax = axes[1, 0]
    for label, (column, color) in series.items():
        # Wealth starts at 1, so the running peak is at least 1.
        ax.plot(when, wealth[column] / wealth[column].cummax().clip(lower=1) - 1,
                color=color, linewidth=2, label=label)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    style(ax, "Drawdown")

    ax = axes[1, 1]
    yearly = (1 + returns[["portfolio", "equal_weight"]]).groupby(when.dt.year).prod() - 1
    x = np.arange(len(yearly))
    for offset, (label, (column, color)) in zip((-0.2, 0.2), series.items()):
        ax.bar(x + offset, yearly[column], width=0.38, color=color, label=label)
    ax.set_xticks(x, yearly.index, rotation=90)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    style(ax, "Calendar-year return")

    # One legend for all panels; colors mean the same thing everywhere.
    fig.suptitle(Path(path).stem.removesuffix("_returns"), x=0.01, ha="left", fontsize=11)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("returns", type=Path, help="<run-folder>/<name>_returns.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.returns.with_name(
        args.returns.name.replace("_returns.csv", "_results.png")
    )
    print(f"Saved {plot_run(args.returns, output)}")
