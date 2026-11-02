import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("model,stage", [("gaussian", "nll"), ("baseline", "mse")])
def test_training_log_and_plot(tmp_path, model, stage):
    root = Path(__file__).resolve().parents[1]
    log = tmp_path / f"{model}.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(root / "demo/ml/train.py"),
            "--model",
            model,
            "--smoke",
            "--epochs",
            "2",
            "--output",
            str(tmp_path / "results.csv"),
            "--log",
            str(log),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["event"] == "experiment_start"
    assert records[-1]["event"] == "experiment_end"
    settings = next(r for r in records if r["event"] == "settings")
    assert settings["pretrain"] is True
    assert settings["variance_floor"] == (1e-8 if model == "gaussian" else None)
    epochs = [r for r in records if r["event"] == "epoch"]
    assert [(r["stage"], r["epoch"]) for r in epochs] == [
        (stage, 1),
        (stage, 2),
        ("e2e", 1),
        ("e2e", 2),
    ]
    for r in epochs:
        assert r["quarter"] == "2014-01-01"
        assert r["epoch_seconds"] > 0
        assert r["train_seconds"] + r["validation_seconds"] == pytest.approx(r["epoch_seconds"])
        assert isinstance(r["train_loss"], float) and isinstance(r["validation_loss"], float)
        assert r["timestamp"].endswith("+00:00")

    spec = importlib.util.spec_from_file_location("plot_loss", root / "demo/ml/plot_loss.py")
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    # Live plotting must tolerate an incomplete final write, without altering the log.
    with log.open("a") as stream:
        stream.write('{"event":')
    image = plot.plot_loss(log)
    assert image.read_bytes().startswith(b"\x89PNG")
