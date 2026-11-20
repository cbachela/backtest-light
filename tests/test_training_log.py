import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "model,stage",
    [("gaussian", "nll"), ("baseline", "mse"), ("gaussian_pto", "nll"), ("equal_weight", None), ("gaussian_ensemble", "nll")],
)
def test_training_log_and_plot(tmp_path, model, stage):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "demo/ml/train.py"),
            "--model",
            "gaussian" if model in {"gaussian_pto", "gaussian_ensemble"} else model,
            *(["--members", "2"] if model == "gaussian_ensemble" else []),
            *(["--no-e2e"] if model == "gaussian_pto" else []),
            "--smoke",
            "--epochs",
            "2",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    folders = list((tmp_path / "outputs").iterdir())
    assert len(folders) == 1
    name = model if model in {"gaussian_pto", "equal_weight"} else f"{model}_e2e"
    if model == "gaussian_ensemble":
        name = "gaussian_e2e_ensemble2"
    assert folders[0].name.endswith(f"_{name}")
    assert folders[0].name[:10].count("-") == 2
    summary = pd.read_csv(folders[0] / f"{name}.csv")
    windows = pd.read_csv(folders[0] / f"{name}_windows.csv")
    assert (folders[0] / f"{name}.npz").is_file() is model.startswith("gaussian")
    assert summary.run_folder.iloc[0] == folders[0].name
    assert {"turnover", "holdings", "effective_n", "max_weight", "git_commit"} <= set(summary)
    assert ("ic" in summary) is (model != "equal_weight")
    assert "risk" not in summary
    assert len(windows) == 1 and windows.window.iloc[0] == "2014-01-01"
    assert windows.turnover.iloc[0] == pytest.approx(summary.turnover.iloc[0])
    assert {"cagr", "sortino_ratio", "ew_tracking_error", "ew_information_ratio",
            "ew_cumulative_relative"} <= set(summary) & set(windows)
    returns = pd.read_csv(folders[0] / f"{name}_returns.csv")
    assert len(returns) == summary.observations.iloc[0]
    assert (folders[0] / f"{name}_results.png").read_bytes().startswith(b"\x89PNG")
    if model == "baseline":
        # Comparison of this run alone still adds the equal-weight benchmark row.
        subprocess.run(
            [sys.executable, str(root / "demo/ml/compare.py"), str(folders[0])],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        )
        (comparison,) = (tmp_path / "outputs").glob("comparison_*")
        table = pd.read_csv(comparison / "comparison.csv")
        assert list(table.run) == [name, "equal_weight"]
        assert table.cagr.iloc[1] == pytest.approx(summary.ew_cagr.iloc[0])
        assert (comparison / "comparison.png").read_bytes().startswith(b"\x89PNG")
    (log,) = folders[0].glob("*.jsonl")
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["event"] == "run_start"
    assert records[-1]["event"] == "run_end" and records[-1]["status"] == "completed"
    start = records[0]
    assert start["command"][0].endswith("train.py") and "git_commit" in start
    assert "dirty" not in start and "risk" not in start
    assert start["pretrain"] is (model != "equal_weight")
    assert start["variance_floor"] == (1e-8 if model.startswith("gaussian") else None)
    assert (start["architecture"] is None) is (model == "equal_weight")
    events = {r["event"] for r in records}
    assert not events & {"epoch_start", "test_start", "settings", "experiment_start"}
    assert all("architecture" not in r for r in records if r["event"] == "window_start")
    (window_end,) = [r for r in records if r["event"] == "window_end"]
    assert window_end["window"] == "2014-01-01" and window_end["holdings"] >= 10
    epochs = [r for r in records if r["event"] == "epoch"]
    expected = [
        (stage, 1),
        (stage, 2),
        ("e2e", 1),
        ("e2e", 2),
    ]
    if model == "equal_weight":
        expected = []
    if model == "gaussian_pto":
        expected = expected[:2]
    if model == "gaussian_ensemble":
        expected *= 2
        assert {r["member"] for r in epochs} == {1, 2}
    else:
        assert all("member" not in r and "seed" not in r for r in records)
    assert [(r["stage"], r["epoch"]) for r in epochs] == expected
    for r in epochs:
        assert r["window"] == "2014-01-01"
        assert r["seconds"] > 0
        assert isinstance(r["train_loss"], float) and isinstance(r["validation_loss"], float)
        assert r["timestamp"].endswith("+00:00")
        # Forecast diagnostics in every stage, on both passes.
        for split in ("train", "validation"):
            assert r[f"{split}_mu_spread"] > 0 and -1 <= r[f"{split}_ic"] <= 1
            assert (f"{split}_sigma_median" in r) is model.startswith("gaussian")
            # Portfolio diagnostics and loss parts exist only once allocations are solved.
            portfolio = {"holdings", "effective_n", "max_weight", "turnover",
                         "return_term", "risk_term", "cost_term"}
            assert portfolio <= {k[len(split) + 1:] for k in r} or r["stage"] != "e2e"
            if r["stage"] != "e2e":
                assert f"{split}_holdings" not in r
        if r["stage"] == "e2e":
            assert ("validation_nll_term" in r) is model.startswith("gaussian")

    if model == "equal_weight":
        assert start["solver"] is None
        return

    spec = importlib.util.spec_from_file_location("plot_loss", root / "demo/ml/plot_loss.py")
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    # Live plotting must tolerate an incomplete final write, without altering the log.
    with log.open("a") as stream:
        stream.write('{"event":')
    image = plot.plot_loss(log)
    assert image.read_bytes().startswith(b"\x89PNG")
