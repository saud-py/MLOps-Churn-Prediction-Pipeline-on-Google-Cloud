"""
utils.py — Shared helpers: config loading, logging setup, model path resolution.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config relative to the project root."""
    root = _project_root()
    full_path = root / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def _project_root() -> Path:
    """Return the project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to the project root."""
    return _project_root() / relative


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, config: Optional[dict] = None) -> logging.Logger:
    """
    Return a logger that writes to both stdout and a log file.
    If config is provided, the log file path is read from config['paths']['log_file'].
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = "logs/pipeline.log"
    if config:
        log_file = config.get("paths", {}).get("log_file", log_file)

    log_path = resolve_path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Model path helpers
# ---------------------------------------------------------------------------

def versioned_model_path(model_dir: str) -> Path:
    """
    Return a timestamped model path, e.g.
        models/model_20260530_220000.pkl
    Handles both absolute paths (/tmp/models/) and relative ones (models/).
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    p = Path(model_dir)
    path = p if p.is_absolute() else resolve_path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"model_{ts}.pkl"


def latest_model_path(model_dir: str) -> Optional[Path]:
    """
    Return the most recently created .pkl file in model_dir, or None if empty.
    Handles both absolute paths (/tmp/models/) and relative ones (models/).
    """
    p = Path(model_dir)
    path = p if p.is_absolute() else resolve_path(model_dir)
    if not path.exists():
        return None
    models = sorted(path.glob("model_*.pkl"))
    return models[-1] if models else None
