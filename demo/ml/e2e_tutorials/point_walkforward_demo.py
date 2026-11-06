"""Run point PTO and E2E through the production walk-forward trainer.

Run from the backtest-light repository root:
    uv run --no-sync python demo/ml/point_walkforward_demo.py

This is a short wiring example on synthetic data.
The smoke setting keeps the first rolling window small: 16 training dates,
8 validation dates, and 8 held-out test dates.
"""

import pandas as pd
from btlight.data.panel import synthetic_panel
from btlight.ml.training.e2e import run


def main():
    panel = synthetic_panel(seed=7, periods=1400, assets=12, features=10)
    print(
        f"Synthetic panel: {len(panel.dates)} dates, {len(panel.assets)} assets, "
        f"{panel.features.shape[-1]} features"
    )
    print(
        "The smoke run uses one chronological window: 16 train dates, "
        "8 validation dates, and 8 test dates."
    )

    results = []
    for name, finetune in (("Point PTO", False), ("Point E2E", True)):
        print(f"\n{name}")

        def show_window(event, **fields):
            if event == "window_start":
                print(f"Selected split: {fields['splits']}")
            elif event == "stage_start":
                print(f"Training stage: {fields['stage']}")

        result = run(
            panel,
            prediction="point",
            pretrain=True,
            finetune=finetune,
            epochs=2,
            first_test="2014-01-01",
            smoke=True,
            log=show_window,
        )
        results.append(result)

    print("\nHeld-out summaries (8 synthetic dates; not performance evidence):")
    print(
        pd.concat(results, ignore_index=True)[
            ["model", "observations", "test_start", "test_end", "annualized_return", "strategy"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
