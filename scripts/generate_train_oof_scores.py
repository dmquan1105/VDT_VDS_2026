"""Export xgb_tuned train OOF scores using the notebook-equivalent pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.supervised_validation import make_stratified_folds
from src.features.supervised_features import build_raw_supervised_features
from src.models.supervised import create_model, estimate_scale_pos_weight
from src.scorecard.pdo import probability_to_pdo_score
from src.utils.io import load_raw_data
from src.utils.woe import WOEEncoder


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "supervised"
OUTPUT_PATH = OUTPUT_DIR / "train_oof_scores.csv"
NON_FEATURE_COLUMNS = ["CustomerID", "FraudFlag", "FraudType", "Churn"]


XGB_TUNED_PARAMS = {
    "learning_rate": 0.0817286751953337,
    "n_estimators": 176,
    "max_depth": 6,
    "min_child_weight": 8.633090307988533,
    "subsample": 0.7122984917310959,
    "colsample_bytree": 0.6241179724286258,
    "reg_alpha": 1.076665495760227,
    "reg_lambda": 0.0015888893828360387,
}


class FoldSafeModelPipeline(BaseEstimator, ClassifierMixin):
    """Same fold-safe preprocessing/model wrapper used in notebooks/supervised.ipynb."""

    def __init__(
        self,
        model_type,
        model_params=None,
        categorical_encoder="woe",
        scale_numeric=True,
        woe_smoothing=0.5,
        woe_min_samples=30,
    ):
        self.model_type = model_type
        self.model_params = model_params
        self.categorical_encoder = categorical_encoder
        self.scale_numeric = scale_numeric
        self.woe_smoothing = woe_smoothing
        self.woe_min_samples = woe_min_samples

    def fit(self, X, y):
        model_params = dict(self.model_params or {})
        if self.model_type == "xgb" and "scale_pos_weight" not in model_params:
            model_params["scale_pos_weight"] = estimate_scale_pos_weight(y)

        self.model_ = create_model(self.model_type, model_params)
        self.feature_cols_ = [col for col in X.columns if col not in NON_FEATURE_COLUMNS]
        X_clean = X[self.feature_cols_]

        self.numeric_cols_ = X_clean.select_dtypes(include=["number", "bool"]).columns.to_list()
        self.categorical_cols_ = X_clean.select_dtypes(exclude=["number", "bool"]).columns.to_list()

        parts = []
        if self.numeric_cols_:
            self.numeric_imputer_ = SimpleImputer(strategy="median")
            X_num = self.numeric_imputer_.fit_transform(X_clean[self.numeric_cols_])
            if self.scale_numeric:
                self.numeric_scaler_ = RobustScaler()
                X_num = self.numeric_scaler_.fit_transform(X_num)
            else:
                self.numeric_scaler_ = None
            parts.append(pd.DataFrame(X_num, columns=self.numeric_cols_, index=X.index))
        else:
            self.numeric_imputer_ = None
            self.numeric_scaler_ = None

        if self.categorical_cols_:
            if self.categorical_encoder != "woe":
                raise ValueError(f"Unsupported categorical_encoder: {self.categorical_encoder}")
            self.categorical_encoder_ = WOEEncoder(
                event_label=1,
                smoothing=self.woe_smoothing,
                min_samples=self.woe_min_samples,
            )
            self.categorical_encoder_.fit(
                X_clean[self.categorical_cols_],
                y,
                columns=self.categorical_cols_,
            )
            parts.append(self.categorical_encoder_.transform(X_clean[self.categorical_cols_]))
        else:
            self.categorical_encoder_ = None

        X_model = pd.concat(parts, axis=1)
        self.feature_names_ = X_model.columns.to_list()
        self.model_.fit(X_model, y)
        self.classes_ = getattr(self.model_, "classes_", np.array([0, 1]))
        return self

    def _transform_features(self, X):
        X_clean = X[self.feature_cols_]
        parts = []

        if self.numeric_cols_:
            X_num = self.numeric_imputer_.transform(X_clean[self.numeric_cols_])
            if self.numeric_scaler_ is not None:
                X_num = self.numeric_scaler_.transform(X_num)
            parts.append(pd.DataFrame(X_num, columns=self.numeric_cols_, index=X.index))

        if self.categorical_cols_:
            parts.append(self.categorical_encoder_.transform(X_clean[self.categorical_cols_]))

        return pd.concat(parts, axis=1)[self.feature_names_]

    def predict_proba(self, X):
        return self.model_.predict_proba(self._transform_features(X))


def _build_notebook_training_data() -> tuple[pd.DataFrame, pd.Series]:
    print("Building raw features on the fly, matching notebooks/supervised.ipynb...")
    raw_data = load_raw_data(DATA_DIR)
    features_df = build_raw_supervised_features(
        customer_df=raw_data["dts_train"],
        sim_events=raw_data["sim_events"],
        device_sessions=raw_data["device_sessions"],
        kyc_records=raw_data["kyc_records"],
        device_catalog=raw_data["device_catalog"],
    )

    dts_train_labels = pd.read_csv(DATA_DIR / "dts_train.csv")
    merged = dts_train_labels[["CustomerID", "FraudFlag"]].merge(
        features_df,
        on="CustomerID",
        how="inner",
    )
    if len(merged) != len(dts_train_labels):
        raise ValueError(f"Expected {len(dts_train_labels)} train rows, got {len(merged)}")
    return merged.drop(columns=["FraudFlag"]), merged["FraudFlag"].astype(int)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = _build_notebook_training_data()
    folds = make_stratified_folds(X, y, n_splits=5, random_state=42)

    pipeline = FoldSafeModelPipeline(
        model_type="xgb",
        model_params=XGB_TUNED_PARAMS,
        categorical_encoder="woe",
        scale_numeric=True,
        woe_smoothing=1.7726118113124238,
        woe_min_samples=96,
    )

    raw_oof = np.zeros(len(y), dtype=float)
    fold_ids = np.zeros(len(y), dtype=int)
    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"Fitting xgb_tuned fold {fold_id}/{len(folds)}...")
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        raw_oof[val_idx] = fold_pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        fold_ids[val_idx] = fold_id

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrated_oof = calibrator.fit_transform(raw_oof, y)
    dts = np.round(probability_to_pdo_score(calibrated_oof, base_score=600, pdo=50)).astype(int)

    out = pd.DataFrame(
        {
            "CustomerID": X["CustomerID"].to_numpy(),
            "model_name": "xgb_tuned",
            "Fold": fold_ids,
            "y_true": y.to_numpy(),
            "raw_p_fraud": raw_oof,
            "P_fraud": calibrated_oof,
            "DTS": dts,
        }
    )
    out.to_csv(OUTPUT_PATH, index=False)

    pr_auc = average_precision_score(y, raw_oof)
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} with {len(out):,} rows")
    print(f"OOF raw PR-AUC: {pr_auc:.6f}")


if __name__ == "__main__":
    main()
