"""Fairness and proxy-bias segment diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


REVIEW_ACTIONS = {"manual_review", "decline_or_block"}


def _yes_or_one(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"yes", "y", "true", "1"}) | (pd.to_numeric(series, errors="coerce") == 1)


def _top_decile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(False, index=series.index)
    return numeric >= numeric.quantile(0.90)


def _low_device_tier(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(False, index=series.index)
    min_val = numeric.min()
    max_val = numeric.max()
    if min_val <= 1 <= max_val:
        return numeric <= 1
    return numeric == min_val


def _segment_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {"all_customers": pd.Series(True, index=df.index)}
    if "NonUSTravel" in df.columns:
        masks["frequent_roaming"] = _yes_or_one(df["NonUSTravel"])
    elif "RoamingCalls" in df.columns:
        masks["frequent_roaming"] = _top_decile(df["RoamingCalls"])
    elif "distinct_country_count" in df.columns:
        country_count = pd.to_numeric(df["distinct_country_count"], errors="coerce")
        masks["frequent_roaming"] = (country_count > 1) | _top_decile(country_count)

    if "port_in_flag" in df.columns:
        masks["recent_port_in"] = pd.to_numeric(df["port_in_flag"], errors="coerce").fillna(0) == 1
    elif "days_since_last_port_in" in df.columns:
        masks["recent_port_in"] = pd.to_numeric(df["days_since_last_port_in"], errors="coerce") <= 90

    for col in ["device_tier", "DeviceTier", "device_tier_min", "device_tier_mean"]:
        if col in df.columns:
            masks["low_device_tier"] = _low_device_tier(df[col])
            break

    if "shared_imei_flag" in df.columns:
        masks["shared_device"] = pd.to_numeric(df["shared_imei_flag"], errors="coerce").fillna(0) == 1
    elif "max_customers_per_imei" in df.columns:
        masks["shared_device"] = pd.to_numeric(df["max_customers_per_imei"], errors="coerce").fillna(0) > 1

    if "MonthlyMinutes" in df.columns:
        masks["high_legit_usage"] = _top_decile(df["MonthlyMinutes"])
    elif "avg_sessions_per_day" in df.columns:
        masks["high_legit_usage"] = _top_decile(df["avg_sessions_per_day"])
    return masks


def build_segment_report(
    scored_df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    label_df: pd.DataFrame | None = None,
    review_col: str = "fraud_action",
) -> pd.DataFrame:
    """Build segment metrics for score, review, and optional fraud labels."""
    required = {"CustomerID", "P_fraud", "DTS", review_col}
    missing = required.difference(scored_df.columns)
    if missing:
        raise ValueError(f"Missing required scored columns: {sorted(missing)}")

    df = scored_df.copy()
    if feature_df is not None and "CustomerID" in feature_df.columns:
        feature_cols = [c for c in feature_df.columns if c != "CustomerID" and c not in df.columns]
        df = df.merge(feature_df[["CustomerID", *feature_cols]], on="CustomerID", how="left")
    if label_df is not None and "CustomerID" in label_df.columns:
        label_cols = [c for c in ["CustomerID", "FraudFlag", "FraudType"] if c in label_df.columns]
        extra_label_cols = [c for c in label_cols if c == "CustomerID" or c not in df.columns]
        df = df.merge(label_df[extra_label_cols], on="CustomerID", how="left")

    total = max(len(df), 1)
    rows: list[dict[str, object]] = []
    for segment, mask in _segment_masks(df).items():
        segment_df = df[mask.fillna(False)]
        count = len(segment_df)
        reviewed = segment_df[review_col].isin(REVIEW_ACTIONS) if count else pd.Series(dtype=bool)
        row: dict[str, object] = {
            "segment": segment,
            "count": count,
            "share": count / total,
            "mean_p_fraud": segment_df["P_fraud"].mean() if count else np.nan,
            "median_p_fraud": segment_df["P_fraud"].median() if count else np.nan,
            "mean_dts": segment_df["DTS"].mean() if count else np.nan,
            "median_dts": segment_df["DTS"].median() if count else np.nan,
            "review_rate": reviewed.mean() if count else np.nan,
            "decline_or_block_rate": (
                (segment_df[review_col] == "decline_or_block").mean() if count else np.nan
            ),
        }
        if "FraudFlag" in segment_df.columns:
            fraud = pd.to_numeric(segment_df["FraudFlag"], errors="coerce")
            reviewed_count = int(reviewed.sum()) if count else 0
            reviewed_fraud_count = int(fraud[reviewed].sum()) if reviewed_count else 0
            nonfraud = fraud == 0
            row.update(
                {
                    "fraud_rate": fraud.mean() if count else np.nan,
                    "reviewed_count": reviewed_count,
                    "reviewed_fraud_count": reviewed_fraud_count,
                    "reviewed_precision": reviewed_fraud_count / reviewed_count if reviewed_count else np.nan,
                    "false_positive_review_rate": (
                        reviewed[nonfraud].mean() if nonfraud.sum() else np.nan
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)
