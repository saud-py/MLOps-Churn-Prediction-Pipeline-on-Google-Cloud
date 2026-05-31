"""
test_validation.py — Unit tests for the data validation layer.

Run:
    cd churn-mlops
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from validate import ValidationError, validate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "required_columns": ["customer_id", "monthly_spend", "login_count", "tenure_days", "churn"],
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
    "paths": {"log_file": "logs/test.log"},
}


def _good_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "monthly_spend": [100.0, 50.0, 200.0],
            "login_count": [30, 3, 45],
            "tenure_days": [800, 40, 1200],
            "churn": [0, 1, 0],
        }
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_dataframe_passes():
    """A clean DataFrame should pass all checks without raising."""
    validate(_good_df(), MINIMAL_CONFIG)


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

def test_missing_column_raises():
    df = _good_df().drop(columns=["login_count"])
    with pytest.raises(ValidationError, match="login_count"):
        validate(df, MINIMAL_CONFIG)


def test_multiple_missing_columns_raises():
    df = _good_df().drop(columns=["login_count", "tenure_days"])
    with pytest.raises(ValidationError):
        validate(df, MINIMAL_CONFIG)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def test_null_values_raise():
    df = _good_df()
    df.loc[0, "monthly_spend"] = None
    with pytest.raises(ValidationError, match="Missing values"):
        validate(df, MINIMAL_CONFIG)


# ---------------------------------------------------------------------------
# Range rules
# ---------------------------------------------------------------------------

def test_negative_tenure_raises():
    df = _good_df()
    df.loc[0, "tenure_days"] = -1
    with pytest.raises(ValidationError, match="tenure_days"):
        validate(df, MINIMAL_CONFIG)


def test_negative_monthly_spend_raises():
    df = _good_df()
    df.loc[1, "monthly_spend"] = -10.0
    with pytest.raises(ValidationError, match="monthly_spend"):
        validate(df, MINIMAL_CONFIG)


def test_churn_out_of_range_raises():
    df = _good_df()
    df.loc[0, "churn"] = 2  # invalid: must be 0 or 1
    with pytest.raises(ValidationError, match="churn"):
        validate(df, MINIMAL_CONFIG)


def test_negative_login_count_raises():
    df = _good_df()
    df.loc[2, "login_count"] = -5
    with pytest.raises(ValidationError, match="login_count"):
        validate(df, MINIMAL_CONFIG)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_dataframe_raises_on_missing_columns():
    """An empty DataFrame still needs the right columns."""
    df = pd.DataFrame()
    with pytest.raises(ValidationError):
        validate(df, MINIMAL_CONFIG)


def test_single_row_valid():
    df = pd.DataFrame(
        {
            "customer_id": [99],
            "monthly_spend": [9.99],
            "login_count": [1],
            "tenure_days": [1],
            "churn": [0],
        }
    )
    validate(df, MINIMAL_CONFIG)  # should not raise
