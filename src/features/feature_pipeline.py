import pandas as pd

from src.features.identity_features import build_identity_features
from src.features.sim_features import build_sim_features
from src.features.device_features import build_device_features
from src.features.behavior_features import build_behavior_features


def build_feature_matrix(
    customer_df: pd.DataFrame,
    sim_events: pd.DataFrame,
    device_sessions: pd.DataFrame,
    kyc_records: pd.DataFrame,
    device_catalog: pd.DataFrame,
    include_customer_features: bool = True,
) -> pd.DataFrame:
    customer_base = customer_df[["CustomerID"]].copy()

    reference_date = max(
        pd.to_datetime(sim_events["EventDate"]).max(),
        pd.to_datetime(device_sessions["SessionDate"]).max(),
    )

    identity_features = build_identity_features(
        kyc_records
    )

    sim_features = build_sim_features(
        sim_events,
        reference_date
    )

    device_features = build_device_features(
        device_sessions,
        device_catalog
    )

    behavior_features = build_behavior_features(
        device_sessions,
        reference_date
    )

    feature_df = (
        customer_base
        .merge(identity_features, on="CustomerID", how="left")
        .merge(sim_features, on="CustomerID", how="left")
        .merge(device_features, on="CustomerID", how="left")
        .merge(behavior_features, on="CustomerID", how="left")
    )

    if include_customer_features:
        label_cols = [
            "FraudFlag",
            "FraudType",
            "Churn",
        ]

        customer_features = customer_df.drop(
            columns=[
                col for col in label_cols
                if col in customer_df.columns
            ],
            errors="ignore"
        )

        feature_df = feature_df.merge(
            customer_features,
            on="CustomerID",
            how="left"
        )

    return feature_df