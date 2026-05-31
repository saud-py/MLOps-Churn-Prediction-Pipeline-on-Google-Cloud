"""
gcs.py — Google Cloud Storage helpers for the churn pipeline.

Handles:
  - Downloading input files (raw data) from GCS to /tmp/
  - Uploading output files (models, predictions) from /tmp/ back to GCS

The pipeline calls these only when the GCS_BUCKET environment variable is set.
Local runs skip GCS entirely and use paths from config.yaml as-is.
"""

import os
from pathlib import Path
from typing import Optional

from utils import get_logger


def gcs_enabled() -> bool:
    """Return True if GCS_BUCKET env var is set (i.e. running in Cloud Run)."""
    return bool(os.environ.get("GCS_BUCKET", "").strip())


def bucket_name() -> str:
    return os.environ["GCS_BUCKET"]


def download_inputs(config: dict, logger=None) -> dict:
    """
    Download raw input files from GCS into /tmp/ and return an updated
    paths dict so the rest of the pipeline uses the local /tmp/ copies.

    GCS layout expected:
        gs://<bucket>/raw/customers.csv
        gs://<bucket>/raw/customers_daily.csv

    Returns
    -------
    dict  Updated paths sub-dict with local /tmp/ paths substituted in.
    """
    if logger is None:
        logger = get_logger("gcs")

    from google.cloud import storage  # lazy import — not needed for local runs

    client = storage.Client()
    bucket = client.bucket(bucket_name())

    paths = dict(config["paths"])  # shallow copy — we'll override some keys

    # Map of GCS blob path → local destination
    downloads = {
        "raw/customers.csv":       "/tmp/customers.csv",
        "raw/customers_daily.csv": "/tmp/customers_daily.csv",
    }

    for blob_name, local_path in downloads.items():
        blob = bucket.blob(blob_name)
        if blob.exists():
            logger.info("GCS ↓  gs://%s/%s  →  %s", bucket_name(), blob_name, local_path)
            blob.download_to_filename(local_path)
        else:
            logger.warning("GCS blob not found: gs://%s/%s — skipping", bucket_name(), blob_name)

    # Override paths to point at /tmp/ copies
    paths["raw_data"]   = "/tmp/customers.csv"
    paths["daily_data"] = "/tmp/customers_daily.csv"
    paths["model_dir"]  = "/tmp/models/"
    paths["prediction_dir"] = "/tmp/predictions/"
    paths["log_file"]   = "/tmp/pipeline.log"

    Path("/tmp/models").mkdir(exist_ok=True)
    Path("/tmp/predictions").mkdir(exist_ok=True)

    return paths


def upload_outputs(config: dict, logger=None) -> None:
    """
    Upload model artifacts and prediction CSVs from /tmp/ back to GCS.

    GCS layout written:
        gs://<bucket>/models/model_<timestamp>.pkl
        gs://<bucket>/predictions/predictions_<timestamp>.csv
    """
    if logger is None:
        logger = get_logger("gcs")

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name())

    upload_dirs = {
        "/tmp/models":      "models",
        "/tmp/predictions": "predictions",
    }

    for local_dir, gcs_prefix in upload_dirs.items():
        for local_file in Path(local_dir).glob("*"):
            if local_file.is_file():
                blob_name = f"{gcs_prefix}/{local_file.name}"
                blob = bucket.blob(blob_name)
                logger.info("GCS ↑  %s  →  gs://%s/%s", local_file, bucket_name(), blob_name)
                blob.upload_from_filename(str(local_file))
