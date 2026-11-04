import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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
    assert (folders[0] / f"{name}.csv").is_file()
    assert (folders[0] / f"{name}.npz").is_file() is model.startswith("gaussian")
    (log,) = folders[0].glob("*.jsonl")
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["event"] == "experiment_start"
    assert records[-1]["event"] == "experiment_end"
    settings = next(r for r in records if r["event"] == "settings")
    assert settings["pretrain"] is (model != "equal_weight")
    assert settings["variance_floor"] == (1e-8 if model.startswith("gaussian") else None)
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
    assert [(r["stage"], r["epoch"]) for r in epochs] == expected
    for r in epochs:
        assert r["quarter"] == "2014-01-01"
        assert r["epoch_seconds"] > 0
        assert r["train_seconds"] + r["validation_seconds"] == pytest.approx(r["epoch_seconds"])
        assert isinstance(r["train_loss"], float) and isinstance(r["validation_loss"], float)
        assert r["timestamp"].endswith("+00:00")

    if model == "equal_weight":
        assert settings["solver"] is None
        return

    spec = importlib.util.spec_from_file_location("plot_loss", root / "demo/ml/plot_loss.py")
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    # Live plotting must tolerate an incomplete final write, without altering the log.
    with log.open("a") as stream:
        stream.write('{"event":')
    image = plot.plot_loss(log)
    assert image.read_bytes().startswith(b"\x89PNG")
