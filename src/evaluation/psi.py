"""Population Stability Index utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _psi_component(expected_pct: pd.Series, actual_pct: pd.Series, eps: float) -> pd.Series:
    expected_safe = expected_pct.clip(lower=eps)
    actual_safe = actual_pct.clip(lower=eps)
    return (actual_safe - expected_safe) * np.log(actual_safe / expected_safe)


def calculate_numeric_psi(
    expected: pd.Series,
    actual: pd.Series,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> tuple[float, pd.DataFrame]:
    """Calculate numeric PSI using quantile bins learned from expected/train."""
    expected_clean = pd.to_numeric(pd.Series(expected), errors="coerce").dropna()
    actual_clean = pd.to_numeric(pd.Series(actual), errors="coerce").dropna()
    if expected_clean.empty or actual_clean.empty:
        detail = pd.DataFrame(
            columns=["bucket", "expected_count", "actual_count", "expected_pct", "actual_pct", "psi_component"]
        )
        return np.nan, detail

    quantiles = np.linspace(0, 1, n_bins + 1)
    breakpoints = np.unique(expected_clean.quantile(quantiles).to_numpy())
    if len(breakpoints) <= 2:
        breakpoints = np.array([expected_clean.min(), expected_clean.max()])
    bins = breakpoints.astype(float)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_bins = pd.cut(expected_clean, bins=bins, include_lowest=True, duplicates="drop")
    actual_bins = pd.cut(actual_clean, bins=bins, include_lowest=True, duplicates="drop")
    categories = expected_bins.cat.categories.union(actual_bins.cat.categories)
    expected_counts = expected_bins.value_counts(sort=False).reindex(categories, fill_value=0)
    actual_counts = actual_bins.value_counts(sort=False).reindex(categories, fill_value=0)

    detail = pd.DataFrame(
        {
            "bucket": expected_counts.index.astype(str),
            "expected_count": expected_counts.to_numpy(),
            "actual_count": actual_counts.to_numpy(),
        }
    )
    detail["expected_pct"] = detail["expected_count"] / max(detail["expected_count"].sum(), 1)
    detail["actual_pct"] = detail["actual_count"] / max(detail["actual_count"].sum(), 1)
    detail["psi_component"] = _psi_component(detail["expected_pct"], detail["actual_pct"], eps)
    return float(detail["psi_component"].sum()), detail


def calculate_categorical_psi(
    expected: pd.Series,
    actual: pd.Series,
    eps: float = 1e-6,
) -> tuple[float, pd.DataFrame]:
    """Calculate PSI for categorical variables using unioned categories."""
    expected_filled = pd.Series(expected).astype("object").where(pd.Series(expected).notna(), "__MISSING__")
    actual_filled = pd.Series(actual).astype("object").where(pd.Series(actual).notna(), "__MISSING__")
    categories = sorted(set(expected_filled.astype(str)).union(set(actual_filled.astype(str))))
    expected_counts = expected_filled.astype(str).value_counts().reindex(categories, fill_value=0)
    actual_counts = actual_filled.astype(str).value_counts().reindex(categories, fill_value=0)
    detail = pd.DataFrame(
        {
            "bucket": categories,
            "expected_count": expected_counts.to_numpy(),
            "actual_count": actual_counts.to_numpy(),
        }
    )
    detail["expected_pct"] = detail["expected_count"] / max(detail["expected_count"].sum(), 1)
    detail["actual_pct"] = detail["actual_count"] / max(detail["actual_count"].sum(), 1)
    detail["psi_component"] = _psi_component(detail["expected_pct"], detail["actual_pct"], eps)
    return float(detail["psi_component"].sum()), detail


def psi_status(psi_value: float) -> str:
    """Map PSI value to a monitoring status."""
    if pd.isna(psi_value):
        return "not_available"
    if psi_value < 0.10:
        return "stable"
    if psi_value < 0.25:
        return "moderate_shift"
    return "significant_shift"


def build_psi_report(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    columns: list[str],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build PSI summary and per-variable detail tables."""
    rows: list[dict[str, object]] = []
    details: dict[str, pd.DataFrame] = {}
    for column in columns:
        if column not in train_df.columns or column not in holdout_df.columns:
            rows.append(
                {
                    "variable": column,
                    "psi": np.nan,
                    "status": "missing_column",
                    "type": "unknown",
                    "missing_train": np.nan,
                    "missing_holdout": np.nan,
                }
            )
            continue

        train_col = train_df[column]
        holdout_col = holdout_df[column]
        is_numeric = pd.api.types.is_numeric_dtype(train_col) and pd.api.types.is_numeric_dtype(holdout_col)
        if is_numeric:
            psi_value, detail = calculate_numeric_psi(train_col, holdout_col, n_bins=n_bins)
            var_type = "numeric"
        else:
            psi_value, detail = calculate_categorical_psi(train_col, holdout_col)
            var_type = "categorical"

        detail.insert(0, "variable", column)
        details[column] = detail
        rows.append(
            {
                "variable": column,
                "psi": psi_value,
                "status": psi_status(psi_value),
                "type": var_type,
                "missing_train": float(train_col.isna().mean()),
                "missing_holdout": float(holdout_col.isna().mean()),
            }
        )
    return pd.DataFrame(rows), details

