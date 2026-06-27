import shap
import numpy as np
import pandas as pd

def compute_shap_values(explaining_pipeline, X_explain: pd.DataFrame) -> np.ndarray:
    """
    Computes and standardizes SHAP values for the given pipeline and input DataFrame.
    """
    inner_model = explaining_pipeline.model_
    
    if explaining_pipeline.model_type in ["gbm", "lgbm", "xgb"]:
        explainer = shap.TreeExplainer(inner_model)
        shap_vals = explainer.shap_values(X_explain)
    else:
        explainer = shap.Explainer(inner_model, X_explain)
        shap_vals = explainer(X_explain).values
        
    if shap_vals is None:
        raise ValueError("SHAP values returned None")
        
    # Standardize format of shap values
    if hasattr(shap_vals, "values"):
        shap_array = shap_vals.values
    else:
        shap_array = shap_vals
        
    if isinstance(shap_array, list):
        # TreeExplainer on binary models could return a list of arrays (one for each class).
        # We want the positive class (class 1).
        shap_array = shap_array[1] if len(shap_array) == 2 else shap_array[0]
        
    if len(shap_array.shape) == 3:
        # Some explainers return shape (n_samples, n_features, n_classes).
        # We want class 1 (index 1).
        shap_array = shap_array[:, :, 1]
        
    return shap_array


def build_global_shap_importance(
    X_explain: pd.DataFrame,
    shap_array: np.ndarray,
    top_n: int | None = 30,
) -> pd.DataFrame:
    """
    Builds a global SHAP importance table from local SHAP values.
    """
    importance_df = pd.DataFrame(
        {
            "feature": X_explain.columns,
            "mean_abs_shap": np.abs(shap_array).mean(axis=0),
            "mean_shap": shap_array.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    importance_df.insert(0, "rank", np.arange(1, len(importance_df) + 1))
    if top_n is not None:
        importance_df = importance_df.head(top_n)

    return importance_df.reset_index(drop=True)
