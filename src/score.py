"""
score.py — Batch scoring job.

Steps:
  1. Load the latest versioned model from models/
  2. Read the daily CSV (data/raw/customers_daily.csv)
  3. Validate the daily data
  4. Generate predictions + churn probabilities
  5. Write timestamped predictions CSV to data/predictions/
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from utils import get_logger, latest_model_path, load_config, resolve_path
from validate import ValidationError, validate


def score(config: dict, logger=None) -> str:
    """
    Run batch scoring.

    Returns
    -------
    str  Path to the written predictions file.
    """
    if logger is None:
        logger = get_logger("score", config)

    # ------------------------------------------------------------------
    # 1. Load latest model
    # ------------------------------------------------------------------
    model_path = latest_model_path(config["paths"]["model_dir"])
    if model_path is None:
        msg = "No trained model found in '{}'. Run train.py first.".format(
            config["paths"]["model_dir"]
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    # ------------------------------------------------------------------
    # 2. Load daily data
    # ------------------------------------------------------------------
    daily = config["paths"]["daily_data"]
    daily_path = Path(daily) if Path(daily).is_absolute() else resolve_path(daily)
    logger.info("Loading daily data from %s", daily_path)
    df = pd.read_csv(daily_path)

    # ------------------------------------------------------------------
    # 3. Validate daily data
    #    Use a scoring-specific config that doesn't require the 'churn' label
    # ------------------------------------------------------------------
    scoring_config = _scoring_config(config)
    try:
        validate(df, scoring_config, logger)
    except ValidationError as exc:
        logger.error("Daily data validation FAILED — aborting scoring: %s", exc)
        raise

    # ------------------------------------------------------------------
    # 4. Generate predictions
    # ------------------------------------------------------------------
    features = config["features"]
    X = df[features]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = df[["customer_id"]].copy()
    results["prediction"] = predictions
    results["churn_probability"] = probabilities.round(4)
    results["model_version"] = model_path.name
    results["scored_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(
        "Scored %d customers  |  predicted churners: %d (%.1f%%)",
        len(results),
        results["prediction"].sum(),
        100 * results["prediction"].mean(),
    )

    # ------------------------------------------------------------------
    # 5. Write predictions
    # ------------------------------------------------------------------
    pred = config["paths"]["prediction_dir"]
    pred_dir = Path(pred) if Path(pred).is_absolute() else resolve_path(pred)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = pred_dir / f"predictions_{ts}.csv"
    results.to_csv(out_path, index=False)
    logger.info("Predictions written → %s", out_path)

    return str(out_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scoring_config(config: dict) -> dict:
    """
    Return a validation config suitable for scoring data (no 'churn' label required).
    """
    import copy
    sc = copy.deepcopy(config)
    # Remove 'churn' from required columns and type/range checks for scoring
    sc["required_columns"] = [c for c in sc.get("required_columns", []) if c != "churn"]
    sc.get("column_types", {}).pop("churn", None)
    sc.get("range_rules", {}).pop("churn", None)
    return sc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    log = get_logger("score", cfg)

    try:
        out = score(cfg, log)
        log.info("Scoring complete. Output: %s", out)
    except Exception as exc:
        log.exception("Scoring FAILED: %s", exc)
        sys.exit(1)
