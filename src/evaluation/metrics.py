import numpy as np
from sklearn.metrics import average_precision_score


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes Precision-Recall Area Under Curve (PR-AUC), also known as Average Precision.
    """
    if len(np.unique(y_true)) <= 1:
        # Handle cases where the split only contains 1 class (unlikely, but good to handle)
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def compute_recall_at_percent(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    percent: float = 0.05
) -> float:
    """
    Computes recall@X% review metric.
    Sort validation customers by predicted fraud score descending,
    review top X% of that fold, and compute recall among true frauds captured.
    """
    idx = np.argsort(y_prob)[::-1]
    n_review = int(np.ceil(percent * len(y_true)))
    top_idx = idx[:n_review]
    true_positives = np.sum(y_true[top_idx])
    total_positives = np.sum(y_true)
    if total_positives == 0:
        return 0.0
    return float(true_positives / total_positives)
