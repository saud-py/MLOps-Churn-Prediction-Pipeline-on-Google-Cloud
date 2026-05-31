"""
test_training.py — Unit tests for the training pipeline.

Run:
    cd churn-mlops
    pytest tests/ -v
"""

import sys
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import train
from utils import latest_model_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dirs(tmp_path):
    """
    Create a self-contained temp directory tree so tests don't touch real data.
    Returns a config dict pointing at the temp paths.
    """
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    pred_dir = tmp_path / "data" / "predictions"
    pred_dir.mkdir(parents=True)
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # Write a small synthetic dataset
    df = pd.DataFrame(
        {
            "customer_id": range(1, 51),
            "monthly_spend": [float(i * 10) for i in range(1, 51)],
            "login_count": list(range(1, 51)),
            "tenure_days": list(range(10, 510, 10)),
            "churn": [i % 2 for i in range(50)],
        }
    )
    csv_path = raw_dir / "customers.csv"
    df.to_csv(csv_path, index=False)

    config = {
        "model": {
            "random_state": 42,
            "test_size": 0.2,
            "n_estimators": 10,  # small for speed
        },
        "features": ["monthly_spend", "login_count", "tenure_days"],
        "target": "churn",
        "required_columns": [
            "customer_id", "monthly_spend", "login_count", "tenure_days", "churn"
        ],
        "column_types": {
            "monthly_spend": "float",
            "login_count": "int",
            "tenure_days": "int",
            "churn": "int",
        },
        "range_rules": {
            "monthly_spend": {"min": 0.0},
            "login_count": {"min": 0},
            "tenure_days": {"min": 0},
            "churn": {"min": 0, "max": 1},
        },
        "paths": {
            "raw_data": str(csv_path),
            "model_dir": str(model_dir) + "/",
            "prediction_dir": str(pred_dir) + "/",
            "log_file": str(logs_dir / "test.log"),
        },
        "mlflow": {
            "experiment_name": "test-churn",
            "tracking_uri": str(mlruns_dir),
        },
    }
    return config, tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_train_creates_model_file(tmp_dirs, monkeypatch):
    """train() must save a .pkl file in the model directory."""
    config, tmp_path = tmp_dirs

    # Patch resolve_path so utils resolves against tmp_path
    import utils as utils_mod
    original_resolve = utils_mod.resolve_path

    def patched_resolve(relative):
        p = Path(relative)
        if p.is_absolute():
            return p
        return p  # already absolute in config

    monkeypatch.setattr(utils_mod, "resolve_path", patched_resolve)

    saved = train(config)
    assert Path(saved).exists(), "Model file was not created"
    assert saved.endswith(".pkl"), "Model file should be a .pkl"


def test_train_produces_versioned_filename(tmp_dirs, monkeypatch):
    """Model filename must follow the model_YYYYMMDD_HHMMSS.pkl pattern."""
    import re
    config, _ = tmp_dirs

    import utils as utils_mod
    monkeypatch.setattr(utils_mod, "resolve_path", lambda p: Path(p))

    saved = train(config)
    filename = Path(saved).name
    pattern = r"^model_\d{8}_\d{6}\.pkl$"
    assert re.match(pattern, filename), f"Unexpected filename format: {filename}"


def test_train_twice_creates_two_artifacts(tmp_dirs, monkeypatch):
    """Each training run should produce a distinct versioned artifact."""
    import time
    config, _ = tmp_dirs

    import utils as utils_mod
    monkeypatch.setattr(utils_mod, "resolve_path", lambda p: Path(p))

    path1 = train(config)
    time.sleep(1)  # ensure different timestamp
    path2 = train(config)

    assert path1 != path2, "Two training runs should produce different artifact paths"
    assert Path(path1).exists()
    assert Path(path2).exists()


def test_latest_model_path_returns_newest(tmp_dirs, monkeypatch):
    """latest_model_path() should return the most recently created model."""
    import time
    config, _ = tmp_dirs

    import utils as utils_mod
    monkeypatch.setattr(utils_mod, "resolve_path", lambda p: Path(p))

    train(config)
    time.sleep(1)
    second = train(config)

    latest = latest_model_path(config["paths"]["model_dir"])
    assert latest is not None
    assert str(latest) == second, "latest_model_path should return the newest artifact"


def test_train_logs_metrics(tmp_dirs, monkeypatch, capsys):
    """Training should log accuracy, f1_score, roc_auc etc. to stdout."""
    config, _ = tmp_dirs

    import utils as utils_mod
    monkeypatch.setattr(utils_mod, "resolve_path", lambda p: Path(p))

    train(config)
    captured = capsys.readouterr()
    for metric in ("accuracy", "f1_score", "roc_auc"):
        assert metric in captured.out or metric in captured.err, (
            f"Expected metric '{metric}' in log output"
        )
