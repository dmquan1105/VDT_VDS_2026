from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except Exception:
    lgb = None
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False


def create_logistic_regression(params: dict | None = None) -> LogisticRegression:
    model_params = {
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42,
        "C": 1.0,
        "penalty": "l2",
    }
    if params:
        model_params.update(params)
    return LogisticRegression(**model_params)


def create_gradient_boosting(params: dict | None = None) -> HistGradientBoostingClassifier:
    model_params = {
        "class_weight": "balanced",
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_depth": 5,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "random_state": 42,
    }
    if params:
        params = params.copy()
        # Map n_estimators to max_iter if provided for sklearn compatibility
        if "n_estimators" in params:
            n_est = params.pop("n_estimators")
            if "max_iter" not in params:
                params["max_iter"] = n_est
        model_params.update(params)
    return HistGradientBoostingClassifier(**model_params)


def create_lightgbm(params: dict | None = None):
    if not LGBM_AVAILABLE:
        raise ImportError(
            "LightGBM is not available or failed to load. On macOS, install OpenMP "
            "with `brew install libomp`, then reinstall/update the Python package."
        )

    model_params = {
        "class_weight": "balanced",
        "learning_rate": 0.05,
        "n_estimators": 100,
        "max_depth": -1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
        "verbosity": -1,
    }
    if params:
        model_params.update(params)
    return lgb.LGBMClassifier(**model_params)


def create_xgboost(params: dict | None = None):
    if not XGB_AVAILABLE:
        raise ImportError(
            "XGBoost is not available or failed to load. On macOS, install OpenMP "
            "with `brew install libomp`, then reinstall/update the Python package."
        )

    model_params = {
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "n_estimators": 100,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
    }
    if params:
        model_params.update(params)
    return xgb.XGBClassifier(**model_params)


def create_model(model_type: str, params: dict | None = None):
    if model_type == "logistic":
        return create_logistic_regression(params)
    if model_type == "gbm":
        return create_gradient_boosting(params)
    if model_type == "lgbm":
        return create_lightgbm(params)
    if model_type == "xgb":
        return create_xgboost(params)
    raise ValueError(f"Unknown supervised model type: {model_type}")


def estimate_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    y_arr = np.asarray(y)
    positives = np.sum(y_arr == 1)
    negatives = np.sum(y_arr == 0)
    if positives == 0:
        return 1.0
    return float(negatives / positives)


def get_lr_coefficients(
    fitted_model: LogisticRegression,
    feature_names: list[str],
) -> pd.DataFrame:
    if not hasattr(fitted_model, "coef_"):
        raise ValueError("LogisticRegression model must be fitted before reading coefficients.")

    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": fitted_model.coef_[0],
        }
    )
    coef_df["AbsCoefficient"] = coef_df["Coefficient"].abs()

    interpretations = []
    for _, row in coef_df.iterrows():
        feature = row["Feature"]
        coefficient = row["Coefficient"]

        if feature.endswith("_woe"):
            if coefficient < 0:
                interpretation = (
                    "Lower/more negative WOE means higher category fraud risk; "
                    "negative coefficient aligns with risk intuition."
                )
            else:
                interpretation = (
                    "Higher/more positive WOE means lower category fraud risk; "
                    "positive coefficient is counter-intuitive for fraud risk."
                )
        elif coefficient > 0:
            interpretation = f"Higher {feature} increases the model fraud score."
        else:
            interpretation = f"Lower {feature} increases the model fraud score."

        interpretations.append(interpretation)

    coef_df["Interpretation"] = interpretations
    return coef_df.sort_values("AbsCoefficient", ascending=False).reset_index(drop=True)
