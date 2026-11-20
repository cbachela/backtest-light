import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name, *args, cwd):
    subprocess.run([sys.executable, str(ROOT / "demo/ml" / name), *args],
                   cwd=cwd, check=True, capture_output=True, text=True)


# Smoke test of the CLI used by `make experiments`: it runs, writes outputs and a usable log.
@pytest.mark.parametrize(
    "args,trains",
    [
        (["--model", "baseline"], True),
        (["--model", "gaussian"], True),
        (["--model", "gaussian", "--no-e2e"], True),
        (["--model", "gaussian", "--members", "2"], True),
        (["--model", "equal_weight"], False),
    ],
)
def test_train_cli(tmp_path, args, trains):
    script("train.py", *args, "--smoke", "--epochs", "2", cwd=tmp_path)
    (folder,) = (tmp_path / "outputs").iterdir()
    (log,) = folder.glob("*.jsonl")
    assert len(pd.read_csv(log.with_suffix(".csv"))) == 1
    assert next(folder.glob("*_results.png")).read_bytes().startswith(b"\x89PNG")

    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["event"] == "run_start"
    assert records[-1]["status"] == "completed"
    assert any(r["event"] == "epoch" for r in records) is trains

    if trains:
        # Live plotting must tolerate a half-written last line.
        spec = importlib.util.spec_from_file_location("plot_loss", ROOT / "demo/ml/plot_loss.py")
        plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot)
        with log.open("a") as stream:
            stream.write('{"event":')
        assert plot.plot_loss(log).read_bytes().startswith(b"\x89PNG")


def test_compare_cli(tmp_path):
    script("train.py", "--model", "baseline", "--smoke", "--epochs", "1", cwd=tmp_path)
    (folder,) = (tmp_path / "outputs").iterdir()
    script("compare.py", str(folder), cwd=tmp_path)
    (comparison,) = (tmp_path / "outputs").glob("comparison_*")
    # The equal-weight benchmark row is added automatically.
    assert len(pd.read_csv(comparison / "comparison.csv")) == 2
    assert (comparison / "comparison.png").is_file()
