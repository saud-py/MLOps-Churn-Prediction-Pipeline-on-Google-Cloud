"""
train.py — Model training with MLflow experiment tracking and versioned artifacts.

Steps:
  1. Load and split data
  2. Train RandomForestClassifier
  3. Log params + metrics to MLflow
  4. Save versioned model artifact  (models/model_YYYYMMDD_HHMMSS.pkl)
"""

import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from utils import get_logger, load_config, resolve_path, versioned_model_path


def train(config: dict, logger=None) -> str:
    """
    Train the churn model.

    Returns
    -------
    str  Path to the saved model artifact.
    """
    if logger is None:
        logger = get_logger("train", config)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    raw = config["paths"]["raw_data"]
    data_path = Path(raw) if Path(raw).is_absolute() else resolve_path(raw)
    logger.info("Loading training data from %s", data_path)
    df = pd.read_csv(data_path)

    features = config["features"]
    target = config["target"]

    X = df[features]
    y = df[target]

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    # ------------------------------------------------------------------
    # 3. MLflow setup
    # ------------------------------------------------------------------
    tracking_uri = resolve_path(config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        # ------------------------------------------------------------------
        # 4. Train model
        # ------------------------------------------------------------------
        n_estimators = config["model"]["n_estimators"]
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
        logger.info("Training RandomForestClassifier (n_estimators=%d) …", n_estimators)
        clf.fit(X_train, y_train)

        # ------------------------------------------------------------------
        # 5. Evaluate
        # ------------------------------------------------------------------
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        }

        for name, value in metrics.items():
            logger.info("  %s: %s", name, value)

        # ------------------------------------------------------------------
        # 6. Log to MLflow
        # ------------------------------------------------------------------
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("features", features)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, artifact_path="model")

        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)

    # ------------------------------------------------------------------
    # 7. Save versioned artifact locally
    # ------------------------------------------------------------------
    model_path = versioned_model_path(config["paths"]["model_dir"])
    joblib.dump(clf, model_path)
    logger.info("Model saved → %s", model_path)

    return str(model_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    log = get_logger("train", cfg)

    try:
        saved_path = train(cfg, log)
        log.info("Training complete. Artifact: %s", saved_path)
    except Exception as exc:
        log.exception("Training FAILED: %s", exc)
        sys.exit(1)
