"""
pipeline.py — End-to-end batch churn scoring pipeline.

Execution flow:
  1. Load config
  2. If GCS_BUCKET is set: download inputs from GCS → /tmp/
  3. Validate training data
  4. Train model if no model exists
  5. Validate daily scoring data
  6. Run batch scoring
  7. If GCS_BUCKET is set: upload model + predictions back to GCS

Run locally (uses data/raw/ from disk):
    cd churn-mlops
    python src/pipeline.py

Run via Docker with GCS (Cloud Run):
    docker run -e GCS_BUCKET=my-bucket churn-pipeline
"""

import os
import sys
from pathlib import Path

import pandas as pd

from utils import get_logger, latest_model_path, load_config, resolve_path
from validate import ValidationError, validate
from train import train
from score import score
from gcs import gcs_enabled, download_inputs, upload_outputs


def run_pipeline() -> None:
    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    config = load_config()
    logger = get_logger("pipeline", config)

    logger.info("=" * 60)
    logger.info("Churn MLOps Pipeline — starting")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 0: Download inputs from GCS (Cloud Run only)
    # ------------------------------------------------------------------
    if gcs_enabled():
        logger.info("[0/4] GCS mode — downloading inputs from gs://%s …",
                    os.environ["GCS_BUCKET"])
        try:
            config["paths"] = download_inputs(config, logger)
        except Exception as exc:
            logger.exception("GCS download FAILED — pipeline aborted.\n%s", exc)
            sys.exit(1)
    else:
        logger.info("[0/4] Local mode — using files from disk")

    # ------------------------------------------------------------------
    # Step 1: Validate training data
    # ------------------------------------------------------------------
    logger.info("[1/4] Validating training data …")

    # resolve_path handles both absolute (/tmp/...) and relative paths
    raw_data_path = config["paths"]["raw_data"]
    train_path = Path(raw_data_path) if Path(raw_data_path).is_absolute() \
        else resolve_path(raw_data_path)

    if not train_path.exists():
        logger.error("Training data not found at %s", train_path)
        if not gcs_enabled():
            logger.error(
                "Tip: make sure data/raw/customers.csv exists, "
                "or set GCS_BUCKET to download from Cloud Storage."
            )
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    try:
        validate(df_train, config, logger)
    except ValidationError as exc:
        logger.error("Training data validation FAILED — pipeline aborted.\n%s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Train if no model exists
    # ------------------------------------------------------------------
    logger.info("[2/4] Checking for existing model …")
    model_path = latest_model_path(config["paths"]["model_dir"])

    if model_path is None:
        logger.info("No model found — starting training …")
        try:
            saved = train(config, logger)
            logger.info("Training complete. Model saved: %s", saved)
        except Exception as exc:
            logger.exception("Training FAILED — pipeline aborted.\n%s", exc)
            sys.exit(1)
    else:
        logger.info("Existing model found: %s — skipping training.", model_path.name)

    # ------------------------------------------------------------------
    # Step 3: Validate daily scoring data
    # ------------------------------------------------------------------
    logger.info("[3/4] Validating daily scoring data …")
    daily_data_path = config["paths"]["daily_data"]
    daily_path = Path(daily_data_path) if Path(daily_data_path).is_absolute() \
        else resolve_path(daily_data_path)

    if not daily_path.exists():
        logger.warning(
            "Daily scoring file not found at %s — skipping scoring step.", daily_path
        )
        _maybe_upload(config, logger)
        logger.info("Pipeline finished (no scoring data available).")
        return

    # ------------------------------------------------------------------
    # Step 4: Batch scoring
    # ------------------------------------------------------------------
    logger.info("[4/4] Running batch scoring …")
    try:
        out_path = score(config, logger)
        logger.info("Scoring complete. Predictions: %s", out_path)
    except (ValidationError, FileNotFoundError) as exc:
        logger.error("Scoring FAILED — pipeline aborted.\n%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during scoring.\n%s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Upload outputs to GCS (Cloud Run only)
    # ------------------------------------------------------------------
    _maybe_upload(config, logger)

    logger.info("=" * 60)
    logger.info("Pipeline finished successfully ✓")
    logger.info("=" * 60)


def _maybe_upload(config: dict, logger) -> None:
    if gcs_enabled():
        logger.info("Uploading outputs to GCS …")
        try:
            upload_outputs(config, logger)
        except Exception as exc:
            # Non-fatal — predictions are still written locally in /tmp/
            logger.warning("GCS upload failed (outputs saved locally): %s", exc)


if __name__ == "__main__":
    run_pipeline()
