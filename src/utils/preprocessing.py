import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

LABEL_COLS = [
    "FraudFlag",
    "FraudType",
    "Churn",
]

def split_features(
    feature_df: pd.DataFrame,
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:
    if drop_cols is None:
        drop_cols = []
        
    cols_to_drop = [
        "CustomerID",
        *LABEL_COLS,
        *drop_cols,
    ]
    
    X = feature_df.drop(
        columns=[
            col for col in cols_to_drop
            if col in feature_df.columns
        ],
        errors="ignore"
    )
    
    return X

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(
        include=["number", "bool"]
    ).columns.to_list()
    
    categorical_cols = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.to_list()
    
    numeric_pipeline = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]
    )
    
    categorical_pipeline = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor

def preprocess_unsupervised_holdout(
    holdout_features: pd.DataFrame,
):
    """
    Track A - Unsupervised:
    Fit preprocessing trên dts_holdout-derived features.
    không dùng label
    """

    X_holdout = split_features(holdout_features)

    preprocessor = build_preprocessor(X_holdout)

    X_holdout_processed = preprocessor.fit_transform(
        X_holdout
    )

    return (
        X_holdout_processed,
        preprocessor,
        X_holdout,
    )