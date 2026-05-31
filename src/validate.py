"""
validate.py — Data validation layer.

Checks performed (in order):
  1. Required columns present
  2. Column types match config
  3. No missing values
  4. Range rules (min / max per column)

Raises ValidationError on the first failure so the pipeline exits early
with a clear, human-readable message instead of producing silent bad predictions.
"""

import pandas as pd

from utils import get_logger, load_config, resolve_path


class ValidationError(Exception):
    """Raised when the input data fails any validation check."""


def validate(df: pd.DataFrame, config: dict, logger=None) -> None:
    """
    Validate *df* against the rules in *config*.

    Parameters
    ----------
    df     : DataFrame to validate
    config : Loaded config dict (from config.yaml)
    logger : Optional logger; one is created if not supplied

    Raises
    ------
    ValidationError  if any check fails
    """
    if logger is None:
        logger = get_logger("validate", config)

    logger.info("Starting data validation (%d rows, %d cols)", *df.shape)

    _check_required_columns(df, config, logger)
    _check_column_types(df, config, logger)
    _check_missing_values(df, logger)
    _check_range_rules(df, config, logger)

    logger.info("Validation passed ✓")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_required_columns(df: pd.DataFrame, config: dict, logger) -> None:
    required = config.get("required_columns", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"Missing required columns: {missing}"
        logger.error(msg)
        raise ValidationError(msg)
    logger.info("  ✓ All required columns present")


def _check_column_types(df: pd.DataFrame, config: dict, logger) -> None:
    type_map = {"float": "float64", "int": "int64"}
    rules = config.get("column_types", {})
    errors = []

    for col, expected_type in rules.items():
        if col not in df.columns:
            continue  # already caught above
        actual = df[col].dtype
        expected_np = type_map.get(expected_type, expected_type)
        # Allow int64 / float64 / int32 / float32 etc.
        if not str(actual).startswith(expected_type):
            errors.append(f"  Column '{col}': expected {expected_type}, got {actual}")

    if errors:
        msg = "Type mismatch(es):\n" + "\n".join(errors)
        logger.error(msg)
        raise ValidationError(msg)
    logger.info("  ✓ Column types valid")


def _check_missing_values(df: pd.DataFrame, logger) -> None:
    missing = df.isnull().sum()
    bad = missing[missing > 0]
    if not bad.empty:
        detail = bad.to_dict()
        msg = f"Missing values detected: {detail}"
        logger.error(msg)
        raise ValidationError(msg)
    logger.info("  ✓ No missing values")


def _check_range_rules(df: pd.DataFrame, config: dict, logger) -> None:
    rules = config.get("range_rules", {})
    errors = []

    for col, bounds in rules.items():
        if col not in df.columns:
            continue
        if "min" in bounds:
            violations = (df[col] < bounds["min"]).sum()
            if violations:
                errors.append(
                    f"  Column '{col}': {violations} value(s) below min={bounds['min']}"
                )
        if "max" in bounds:
            violations = (df[col] > bounds["max"]).sum()
            if violations:
                errors.append(
                    f"  Column '{col}': {violations} value(s) above max={bounds['max']}"
                )

    if errors:
        msg = "Range rule violation(s):\n" + "\n".join(errors)
        logger.error(msg)
        raise ValidationError(msg)
    logger.info("  ✓ Range rules satisfied")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cfg = load_config()
    log = get_logger("validate", cfg)

    data_path = resolve_path(cfg["paths"]["raw_data"])
    log.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    try:
        validate(df, cfg, log)
    except ValidationError as exc:
        log.error("Validation FAILED: %s", exc)
        sys.exit(1)
