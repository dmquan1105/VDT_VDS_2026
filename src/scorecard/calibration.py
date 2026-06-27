import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def fit_isotonic_calibrator(raw_probs: np.ndarray, true_labels: np.ndarray) -> IsotonicRegression:
    """
    Fits an IsotonicRegression calibrator using OOF raw predictions and true labels.
    """
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(raw_probs, true_labels)
    return iso_reg

def build_reliability_curve(
    true_y: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Produces a reliability curve DataFrame with bins.
    """
    df_cal = pd.DataFrame({
        "y_true": true_y,
        "y_prob_raw": raw_probs,
        "y_prob_calibrated": calibrated_probs
    })
    df_cal["bin"] = pd.cut(df_cal["y_prob_raw"], bins=bins, include_lowest=True)
    reliability_df = df_cal.groupby("bin", observed=False).agg(
        bin_count=("y_true", "count"),
        mean_predicted_prob=("y_prob_raw", "mean"),
        mean_calibrated_prob=("y_prob_calibrated", "mean"),
        actual_fraud_rate=("y_true", "mean")
    ).reset_index()
    reliability_df.rename(columns={"bin": "Bin_Range"}, inplace=True)
    reliability_df["Bin_Range"] = reliability_df["Bin_Range"].astype(str)
    return reliability_df
