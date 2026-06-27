from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.evaluation.metrics import compute_pr_auc, compute_recall_at_percent


def make_stratified_folds(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Creates stratified K-fold splits based on the binary target (FraudFlag).

    Parameters:
    -----------
    features_df: pd.DataFrame
        The input feature DataFrame.
    target_series: pd.Series
        The target label Series (FraudFlag).
    n_splits: int
        Number of splits (default 5).
    random_state: int
        Random seed for reproducibility.

    Returns:
    --------
    list of tuple of (train_indices, val_indices)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(features_df, target_series))
    return folds


def evaluate_fold_predictions(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    percent: float = 0.05,
) -> dict[str, float]:
    """
    Evaluates predictions for a fold using PR-AUC and recall@X% review rate.

    Parameters:
    -----------
    y_true: np.ndarray or pd.Series
        True labels.
    y_prob: np.ndarray or pd.Series
        Predicted probabilities.
    percent: float
        The operational review rate (default 0.05, i.e., 5%).

    Returns:
    --------
    dict
        Dictionary containing calculated metrics:
        - "pr_auc": Precision-Recall AUC
        - f"recall_at_{int(percent*100)}%": Recall at the specified review rate
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    pr_auc = compute_pr_auc(y_true_arr, y_prob_arr)
    recall_at_review = compute_recall_at_percent(y_true_arr, y_prob_arr, percent=percent)

    return {
        "pr_auc": pr_auc,
        f"recall_at_{int(percent*100)}%": recall_at_review,
    }