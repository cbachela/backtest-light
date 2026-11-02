"""Plot training/validation losses from a completed or still-running JSONL log."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss(path, output=None):
    path = Path(path)
    records = []
    for line in path.read_text().splitlines(keepends=True):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if not line.endswith("\n"):
                break  # A running process may still be writing the last line.
            raise
        if record.get("event") == "epoch":
            records.append(record)
    if not records:
        raise ValueError("No completed epochs in this log yet")
    quarters = list(dict.fromkeys(r["quarter"] for r in records))
    stages = list(dict.fromkeys(r["stage"] for r in records))
    fig, axes = plt.subplots(
        len(quarters), len(stages), squeeze=False, figsize=(6 * len(stages), 3.5 * len(quarters))
    )
    for row, quarter in enumerate(quarters):
        for col, stage in enumerate(stages):
            ax = axes[row, col]
            values = [r for r in records if r["quarter"] == quarter and r["stage"] == stage]
            if not values:
                ax.set_visible(False)
                continue
            epochs = [r["epoch"] for r in values]
            ax.plot(epochs, [r["train_loss"] for r in values], ".-", label="Training")
            ax.plot(epochs, [r["validation_loss"] for r in values], ".-", label="Validation")
            ax.set(title=f"{quarter} · {stage.upper()}", xlabel="Epoch", ylabel="Loss")
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle(path.stem)
    fig.tight_layout()
    output = Path(output) if output else path.with_suffix(".loss.png")
    if output.resolve() == path.resolve():
        raise ValueError("plot output must not overwrite the training log")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(f"Saved {plot_loss(args.log, args.output)}")
