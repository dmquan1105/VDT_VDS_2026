from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_errors_by_fraud_type(
    oof_df: pd.DataFrame,
    train_labels_df: pd.DataFrame,
    review_rate: float = 0.05,
    include_non_fraud: bool = False,
) -> pd.DataFrame:
    """
    Analyzes model out-of-fold predictions by FraudType.

    Parameters:
    -----------
    oof_df: pd.DataFrame
        Out-of-fold predictions. Must contain 'CustomerID' and 'y_prob'.
    train_labels_df: pd.DataFrame
        DataFrame containing training labels. Must contain 'CustomerID', 'FraudFlag' and 'FraudType'.
    review_rate: float
        The operational review rate (default 0.05 for top 5% review).
    include_non_fraud: bool
        Whether to include non-fraud transactions (FraudFlag == 0) in the final output.
        Defaults to False (only analyzing fraud archetypes).

    Returns:
    --------
    pd.DataFrame
        Analysis table containing columns:
        - FraudType
        - FraudCount
        - CapturedFraudCount
        - RecallAtReviewRate
        - MeanScore
        - MeanRankPct
        - Priority
    """
    # 1. Align predictions and labels
    merged = oof_df[["CustomerID", "y_prob"]].merge(
        train_labels_df[["CustomerID", "FraudFlag", "FraudType"]],
        on="CustomerID",
        how="inner",
    )

    # 2. Fill missing FraudType for normal transactions
    merged["FraudType"] = merged["FraudType"].fillna("none")

    # 3. Determine the threshold at the specified top X% review rate (over all transactions)
    n_review = int(np.ceil(review_rate * len(merged)))
    # Sort descending and get the score at rank n_review
    if n_review > 0:
        threshold = merged["y_prob"].nlargest(n_review).min()
    else:
        threshold = float("inf")

    # 4. Label who got reviewed
    merged["IsReviewed"] = merged["y_prob"] >= threshold

    # 5. Compute percentile rank for each transaction (high score = high percentile)
    merged["RankPct"] = merged["y_prob"].rank(pct=True)

    # 6. Filter by FraudFlag if not including non-fraud
    if not include_non_fraud:
        analysis_df = merged[merged["FraudFlag"] == 1].copy()
    else:
        analysis_df = merged.copy()

    if len(analysis_df) == 0:
        return pd.DataFrame(
            columns=["FraudType", "FraudCount", "CapturedFraudCount", "RecallAtReviewRate", "MeanScore", "MeanRankPct", "Priority"]
        )

    # 7. Group by FraudType to calculate metrics
    summary = (
        analysis_df.groupby("FraudType")
        .agg(
            FraudCount=("CustomerID", "count"),
            MeanScore=("y_prob", "mean"),
            CapturedFraudCount=("IsReviewed", "sum"),
            MeanRankPct=("RankPct", "mean"),
        )
        .reset_index()
    )

    summary["RecallAtReviewRate"] = summary["CapturedFraudCount"] / summary["FraudCount"]

    # 8. Add Priority flag for key fraud archetypes (sim_swap_ato, mule)
    summary["Priority"] = summary["FraudType"].isin(["sim_swap_ato", "mule"]).astype(int)

    # 9. Sort by Priority desc, then FraudCount desc
    summary = (
        summary.sort_values(by=["Priority", "FraudCount"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # Ensure exact column order
    cols_order = ["FraudType", "FraudCount", "CapturedFraudCount", "RecallAtReviewRate", "MeanScore", "MeanRankPct", "Priority"]
    summary = summary[cols_order]

    return summary
