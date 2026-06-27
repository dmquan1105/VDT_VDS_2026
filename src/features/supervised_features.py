import pandas as pd
from pathlib import Path
from src.utils.io import load_raw_data
from src.utils.woe import WOEEncoder
from src.features.feature_pipeline import build_feature_matrix


LABEL_COLS = ["FraudFlag", "FraudType", "Churn"]
ID_COL = "CustomerID"

def build_supervised_sim_features(sim_events: pd.DataFrame, reference_date) -> pd.DataFrame:
    df = sim_events.copy()
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    reference_date = pd.to_datetime(reference_date)
    
    swap_df = df[df["EventType"] == "sim_swap"].copy()

    cutoff_7d = reference_date - pd.Timedelta(days=7)
    cutoff_30d = reference_date - pd.Timedelta(days=30)

    sim_swap_count_7d = (
        swap_df[swap_df["EventDate"] >= cutoff_7d]
        .groupby("CustomerID")
        .size()
        .rename("sim_swap_count_7d")
        .reset_index()
    )

    sim_swap_count_30d = (
        swap_df[swap_df["EventDate"] >= cutoff_30d]
        .groupby("CustomerID")
        .size()
        .rename("sim_swap_count_30d")
        .reset_index()
    )
    
    customer_base = df[["CustomerID"]].drop_duplicates()
    features = (
        customer_base
        .merge(sim_swap_count_7d, on="CustomerID", how="left")
        .merge(sim_swap_count_30d, on="CustomerID", how="left")
    )
    features["sim_swap_count_7d"] = features["sim_swap_count_7d"].fillna(0).astype(int)
    features["sim_swap_count_30d"] = features["sim_swap_count_30d"].fillna(0).astype(int)
    return features

def build_supervised_behavior_features(device_sessions: pd.DataFrame, reference_date) -> pd.DataFrame:
    df = device_sessions.copy()
    df["SessionDate"] = pd.to_datetime(df["SessionDate"])
    reference_date = pd.to_datetime(reference_date)

    customer_base = df[["CustomerID"]].drop_duplicates()

    cutoff_7d = reference_date - pd.Timedelta(days=7)
    recent_df_7d = df[df["SessionDate"] >= cutoff_7d].copy()

    distinct_ip_7d = (
        recent_df_7d.groupby("CustomerID")["IP"]
        .nunique()
        .rename("distinct_ip_7d")
        .reset_index()
    )

    distinct_country_7d = (
        recent_df_7d.groupby("CustomerID")["CountryCode"]
        .nunique()
        .rename("distinct_country_7d")
        .reset_index()
    )

    recent_df_7d["is_datacenter"] = (
        recent_df_7d["IPType"] == "datacenter"
    ).astype(int)

    recent_df_7d["is_vpn_proxy"] = (
        recent_df_7d["IPType"] == "vpn_proxy"
    ).astype(int)

    iptype_features_7d = (
        recent_df_7d.groupby("CustomerID")
        .agg(
            datacenter_ratio_7d=("is_datacenter", "mean"),
            vpn_proxy_ratio_7d=("is_vpn_proxy", "mean"),
        )
        .reset_index()
    )

    iptype_features_7d["non_residential_ratio_7d"] = (
        iptype_features_7d["datacenter_ratio_7d"]
        + iptype_features_7d["vpn_proxy_ratio_7d"]
    )

    home_cell_ratio_7d = (
        recent_df_7d.groupby("CustomerID")["IsHomeCell"]
        .mean()
        .rename("home_cell_ratio_7d")
        .reset_index()
    )

    recent_df_7d["night_session"] = (
        (recent_df_7d["SessionHour"] >= 22)
        | (recent_df_7d["SessionHour"] <= 4)
    ).astype(int)

    night_ratio_7d = (
        recent_df_7d.groupby("CustomerID")["night_session"]
        .mean()
        .rename("night_session_ratio_7d")
        .reset_index()
    )

    active_days_7d = (
        recent_df_7d.groupby("CustomerID")["SessionDate"]
        .nunique()
        .rename("active_days_7d")
        .reset_index()
    )

    session_count_7d = (
        recent_df_7d.groupby("CustomerID")
        .size()
        .rename("total_sessions_7d")
        .reset_index()
    )

    avg_sessions_7d = session_count_7d.merge(
        active_days_7d,
        on="CustomerID",
        how="left"
    )

    avg_sessions_7d["avg_sessions_per_day_7d"] = (
        avg_sessions_7d["total_sessions_7d"]
        / avg_sessions_7d["active_days_7d"]
    )

    avg_sessions_7d = avg_sessions_7d[
        ["CustomerID", "total_sessions_7d", "avg_sessions_per_day_7d"]
    ]

    daily_country_count_7d = (
        recent_df_7d.groupby(["CustomerID", "SessionDate"])["CountryCode"]
        .nunique()
        .reset_index(name="country_count")
    )

    daily_country_count_7d["geo_velocity_day"] = (
        daily_country_count_7d["country_count"] > 1
    ).astype(int)

    geo_velocity_7d = (
        daily_country_count_7d
        .groupby("CustomerID")
        .agg(
            geo_velocity_alerts_7d=("geo_velocity_day", "sum"),
            geo_velocity_flag_7d=("geo_velocity_day", "max"),
        )
        .reset_index()
    )

    features = (
        customer_base
        .merge(distinct_ip_7d, on="CustomerID", how="left")
        .merge(distinct_country_7d, on="CustomerID", how="left")
        .merge(iptype_features_7d, on="CustomerID", how="left")
        .merge(home_cell_ratio_7d, on="CustomerID", how="left")
        .merge(night_ratio_7d, on="CustomerID", how="left")
        .merge(active_days_7d, on="CustomerID", how="left")
        .merge(avg_sessions_7d, on="CustomerID", how="left")
        .merge(geo_velocity_7d, on="CustomerID", how="left")
    )

    fill_zero_cols = [
        "distinct_ip_7d",
        "distinct_country_7d",
        "datacenter_ratio_7d",
        "vpn_proxy_ratio_7d",
        "non_residential_ratio_7d",
        "home_cell_ratio_7d",
        "night_session_ratio_7d",
        "active_days_7d",
        "total_sessions_7d",
        "avg_sessions_per_day_7d",
        "geo_velocity_alerts_7d",
        "geo_velocity_flag_7d",
    ]

    for col in fill_zero_cols:
        features[col] = features[col].fillna(0)

    return features


def get_categorical_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [
        col for col in feature_df.select_dtypes(
            include=["object", "string", "category"]
        ).columns.to_list()
        if col != ID_COL and col not in LABEL_COLS
    ]


def add_woe_encoded_features(
    train_features: pd.DataFrame,
    holdout_features: pd.DataFrame,
    dts_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    categorical_cols = get_categorical_feature_columns(train_features)
    if not categorical_cols:
        empty_mapping = pd.DataFrame(
            columns=[
                "feature_name",
                "category",
                "total_count",
                "event_count",
                "non_event_count",
                "event_rate",
                "woe",
                "is_rare_bucket",
                "is_missing_bucket",
            ]
        )
        return train_features, holdout_features, empty_mapping, []

    target = (
        train_features[[ID_COL]]
        .merge(dts_train[[ID_COL, "FraudFlag"]], on=ID_COL, how="left")
        ["FraudFlag"]
    )
    if target.isna().any():
        raise ValueError("Missing FraudFlag while fitting WOE encoder")

    encoder = WOEEncoder(event_label=1, smoothing=0.5, min_samples=30)
    train_woe = encoder.fit_transform(
        train_features[categorical_cols],
        target,
        columns=categorical_cols,
    )
    holdout_woe = encoder.transform(holdout_features[categorical_cols])

    train_features = pd.concat(
        [train_features.reset_index(drop=True), train_woe.reset_index(drop=True)],
        axis=1,
    )
    holdout_features = pd.concat(
        [holdout_features.reset_index(drop=True), holdout_woe.reset_index(drop=True)],
        axis=1,
    )

    woe_cols = train_woe.columns.to_list()
    mapping_df = encoder.mapping_frame_.copy()
    mapping_df["encoded_feature_name"] = mapping_df["feature_name"] + encoder.suffix
    return train_features, holdout_features, mapping_df, woe_cols

def build_raw_supervised_features(
    customer_df: pd.DataFrame,
    sim_events: pd.DataFrame,
    device_sessions: pd.DataFrame,
    kyc_records: pd.DataFrame,
    device_catalog: pd.DataFrame,
    reference_date=None,
) -> pd.DataFrame:
    """
    Builds the raw supervised features (both numerical and categorical)
    for a given customer cohort, before applying WOE encoding.
    """
    if reference_date is None:
        reference_date = max(
            pd.to_datetime(sim_events["EventDate"]).max(),
            pd.to_datetime(device_sessions["SessionDate"]).max(),
        )
    else:
        reference_date = pd.to_datetime(reference_date)

    features = build_feature_matrix(
        customer_df=customer_df,
        sim_events=sim_events,
        device_sessions=device_sessions,
        kyc_records=kyc_records,
        device_catalog=device_catalog,
        include_customer_features=True,
    )

    sim_7d_30d = build_supervised_sim_features(sim_events, reference_date)
    beh_7d = build_supervised_behavior_features(device_sessions, reference_date)

    features = features.merge(sim_7d_30d, on="CustomerID", how="left")
    features = features.merge(beh_7d, on="CustomerID", how="left")

    for col in ["sim_swap_count_7d", "sim_swap_count_30d"]:
        features[col] = features[col].fillna(0).astype(int)
    for col in [
        "distinct_ip_7d", "distinct_country_7d", "datacenter_ratio_7d", "vpn_proxy_ratio_7d",
        "non_residential_ratio_7d", "home_cell_ratio_7d", "night_session_ratio_7d",
        "active_days_7d", "total_sessions_7d", "avg_sessions_per_day_7d",
        "geo_velocity_alerts_7d", "geo_velocity_flag_7d"
    ]:
        features[col] = features[col].fillna(0)

    return features


def generate_supervised_features(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs/supervised"
) -> None:
    """
    Generates supervised feature matrices for train and holdout snapshots,
    ensuring identical schemas, dtypes, and no target leakage.
    Saves outputs to the specified output directory.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw datasets...")
    raw_data = load_raw_data(data_dir)
    
    dts_train = raw_data["dts_train"]
    dts_holdout = raw_data["dts_holdout"]
    sim_events = raw_data["sim_events"]
    device_sessions = raw_data["device_sessions"]
    kyc_records = raw_data["kyc_records"]
    device_catalog = raw_data["device_catalog"]

    reference_date = max(
        pd.to_datetime(sim_events["EventDate"]).max(),
        pd.to_datetime(device_sessions["SessionDate"]).max(),
    )

    print("Building train supervised features...")
    train_features = build_raw_supervised_features(
        customer_df=dts_train,
        sim_events=sim_events,
        device_sessions=device_sessions,
        kyc_records=kyc_records,
        device_catalog=device_catalog,
        reference_date=reference_date,
    )

    print("Building holdout supervised features...")
    holdout_features = build_raw_supervised_features(
        customer_df=dts_holdout,
        sim_events=sim_events,
        device_sessions=device_sessions,
        kyc_records=kyc_records,
        device_catalog=device_catalog,
        reference_date=reference_date,
    )

    print("Fitting WOE encoder on train categorical features...")
    (
        train_features,
        holdout_features,
        woe_mapping,
        woe_cols,
    ) = add_woe_encoded_features(
        train_features=train_features,
        holdout_features=holdout_features,
        dts_train=dts_train,
    )

    # Validate row counts
    assert len(train_features) == 51047, f"Expected 51047 rows in train, got {len(train_features)}"
    assert len(holdout_features) == 20000, f"Expected 20000 rows in holdout, got {len(holdout_features)}"
    assert train_features["CustomerID"].is_unique, "CustomerID is not unique in train features"
    assert holdout_features["CustomerID"].is_unique, "CustomerID is not unique in holdout features"

    # Define exact columns order
    ordered_cols = [
        'CustomerID', 'kyc_level_ord', 'has_face_score', 'has_iddoc_score', 'face_match_score', 'id_doc_match_score',
        'phone_number_age_days', 'sim_swap_count_total', 'sim_swap_count_7d', 'sim_swap_count_30d', 'sim_swap_count_90d',
        'sim_swap_count_12m', 'days_since_last_sim_swap', 'iccid_count', 'port_in_flag', 'recent_sim_change_flag',
        'num_imeis_90d', 'max_customers_per_imei', 'num_accounts_linked_to_device', 'shared_imei_flag', 'high_shared_imei_flag',
        'rooted_session_ratio', 'is_rooted', 'observed_device_days', 'device_tier_mean', 'device_tier_min', 'device_tier_max',
        'low_tier_session_ratio', 'low_tier_device_flag', 'device_catalog_missing_ratio', 'device_catalog_missing_any',
        'is_emulator', 'emulator_session_ratio', 'is_generic_or_clone', 'is_feature_phone', 'tac_customer_count_max',
        'tac_imei_count_max', 'tac_customer_per_imei_max', 'tac_grey_clone_flag', 'tac_risk_score', 'distinct_ip_count',
        'distinct_country_count', 'datacenter_ratio', 'vpn_proxy_ratio', 'non_residential_ratio', 'home_cell_ratio',
        'night_session_ratio', 'active_days_90d', 'total_sessions', 'avg_sessions_per_day', 'geo_velocity_alerts',
        'geo_velocity_flag', 'days_since_first_seen', 'distinct_ip_7d', 'distinct_country_7d', 'datacenter_ratio_7d',
        'vpn_proxy_ratio_7d', 'non_residential_ratio_7d', 'home_cell_ratio_7d', 'night_session_ratio_7d', 'active_days_7d',
        'total_sessions_7d', 'avg_sessions_per_day_7d', 'geo_velocity_alerts_7d', 'geo_velocity_flag_7d', 'distinct_ip_30d',
        'distinct_country_30d', 'datacenter_ratio_30d', 'vpn_proxy_ratio_30d', 'non_residential_ratio_30d',
        'home_cell_ratio_30d', 'night_session_ratio_30d', 'active_days_30d', 'total_sessions_30d', 'avg_sessions_per_day_30d',
        'geo_velocity_alerts_30d', 'geo_velocity_flag_30d', 'MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge',
        'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls', 'PercChangeMinutes', 'PercChangeRevenues', 'DroppedCalls',
        'BlockedCalls', 'UnansweredCalls', 'CustomerCareCalls', 'ThreewayCalls', 'ReceivedCalls', 'OutboundCalls', 'InboundCalls',
        'PeakCallsInOut', 'OffPeakCallsInOut', 'DroppedBlockedCalls', 'CallForwardingCalls', 'CallWaitingCalls', 'MonthsInService',
        'UniqueSubs', 'ActiveSubs', 'ServiceArea', 'Handsets', 'HandsetModels', 'CurrentEquipmentDays', 'AgeHH1', 'AgeHH2',
        'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner', 'RVOwner', 'Homeownership', 'BuysViaMailOrder',
        'RespondsToMailOffers', 'OptOutMailings', 'NonUSTravel', 'OwnsComputer', 'HasCreditCard', 'RetentionCalls',
        'RetentionOffersAccepted', 'NewCellphoneUser', 'NotNewCellphoneUser', 'ReferralsMadeBySubscriber', 'IncomeGroup',
        'OwnsMotorcycle', 'AdjustmentsToCreditRating', 'HandsetPrice', 'MadeCallToRetentionTeam', 'CreditRating', 'PrizmCode',
        'Occupation', 'MaritalStatus'
    ]

    missing_ordered_cols = [
        col for col in ordered_cols
        if col not in train_features.columns or col not in holdout_features.columns
    ]
    if missing_ordered_cols:
        raise ValueError(f"Missing expected supervised feature columns: {missing_ordered_cols}")

    ordered_cols = ordered_cols + woe_cols

    # Reorder columns to match exactly
    train_features = train_features[ordered_cols]
    holdout_features = holdout_features[ordered_cols]

    # Align dtypes
    for col in train_features.columns:
        if train_features[col].dtype != holdout_features[col].dtype:
            print(f"Aligning dtype for column '{col}': {train_features[col].dtype} vs {holdout_features[col].dtype}")
            if pd.api.types.is_numeric_dtype(train_features[col]) and pd.api.types.is_numeric_dtype(holdout_features[col]):
                train_features[col] = train_features[col].astype(float)
                holdout_features[col] = holdout_features[col].astype(float)
            else:
                train_features[col] = train_features[col].astype(str)
                holdout_features[col] = holdout_features[col].astype(str)

    # Save to parquet
    print("Saving feature snapshots...")
    train_path = output_dir / "train_features_supervised.parquet"
    holdout_path = output_dir / "holdout_features_supervised.parquet"
    woe_mapping_path = output_dir / "supervised_woe_mapping.csv"
    
    train_features.to_parquet(train_path, index=False)
    holdout_features.to_parquet(holdout_path, index=False)
    woe_mapping.to_csv(woe_mapping_path, index=False)
    print(f"Saved train features to {train_path} (shape: {train_features.shape})")
    print(f"Saved holdout features to {holdout_path} (shape: {holdout_features.shape})")
    print(f"Saved WOE mapping to {woe_mapping_path}")

    # Generate schema description CSV
    print("Generating schema CSV...")
    schema_rows = []
    
    identity_cols = ["kyc_level_ord", "has_face_score", "has_iddoc_score", "face_match_score", "id_doc_match_score"]
    
    sim_cols = [
        "phone_number_age_days", "sim_swap_count_total", "sim_swap_count_7d", "sim_swap_count_30d",
        "sim_swap_count_90d", "sim_swap_count_12m", "days_since_last_sim_swap", "iccid_count",
        "port_in_flag", "recent_sim_change_flag"
    ]
    
    device_cols = [
        "num_imeis_90d", "max_customers_per_imei", "num_accounts_linked_to_device", "shared_imei_flag",
        "high_shared_imei_flag", "rooted_session_ratio", "is_rooted", "observed_device_days",
        "device_tier_mean", "device_tier_min", "device_tier_max", "low_tier_session_ratio",
        "low_tier_device_flag", "device_catalog_missing_ratio", "device_catalog_missing_any",
        "is_emulator", "emulator_session_ratio", "is_generic_or_clone", "is_feature_phone",
        "tac_customer_count_max", "tac_imei_count_max", "tac_customer_per_imei_max",
        "tac_grey_clone_flag", "tac_risk_score"
    ]
    
    behavior_cols = [
        "distinct_ip_count", "distinct_country_count", "datacenter_ratio", "vpn_proxy_ratio",
        "non_residential_ratio", "home_cell_ratio", "night_session_ratio", "active_days_90d",
        "total_sessions", "avg_sessions_per_day", "geo_velocity_alerts", "geo_velocity_flag",
        "days_since_first_seen", "distinct_ip_7d", "distinct_country_7d", "datacenter_ratio_7d",
        "vpn_proxy_ratio_7d", "non_residential_ratio_7d", "home_cell_ratio_7d", "night_session_ratio_7d",
        "active_days_7d", "total_sessions_7d", "avg_sessions_per_day_7d", "geo_velocity_alerts_7d",
        "geo_velocity_flag_7d", "distinct_ip_30d", "distinct_country_30d", "datacenter_ratio_30d",
        "vpn_proxy_ratio_30d", "non_residential_ratio_30d", "home_cell_ratio_30d", "night_session_ratio_30d",
        "active_days_30d", "total_sessions_30d", "avg_sessions_per_day_30d", "geo_velocity_alerts_30d",
        "geo_velocity_flag_30d"
    ]

    raw_categorical_cols = set(get_categorical_feature_columns(train_features))
    woe_cols_set = set(woe_cols)

    for col in train_features.columns:
        dtype_str = str(train_features[col].dtype)
        
        if col == "CustomerID":
            role = "ID / Entity Key"
            input_status = "Exclude (ID Column)"
            notes = "Unique identifier for each customer"
        elif col in woe_cols_set:
            role = "WOE Encoded Categorical"
            input_status = "Input Feature"
            source_col = col.removesuffix("_woe")
            notes = f"WOE encoding fitted on train FraudFlag for raw categorical feature {source_col}"
        elif col in raw_categorical_cols:
            role = "Raw Categorical"
            input_status = "Exclude Raw (Use WOE)"
            notes = "Raw categorical retained for traceability; use corresponding *_woe feature for supervised model input"
        elif col in identity_cols:
            role = "Identity Confidence"
            input_status = "Input Feature"
            notes = "Derived from KYC records"
        elif col in sim_cols:
            role = "SIM Stability"
            input_status = "Input Feature"
            notes = "Derived from SIM events history"
        elif col in device_cols:
            role = "Device Integrity"
            input_status = "Input Feature"
            notes = "Derived from device sessions and device catalog"
        elif col in behavior_cols:
            role = "Behavioral Consistency"
            input_status = "Input Feature"
            notes = "Derived from session activity behavior and geographic/network stability"
        else:
            role = "Customer Profile"
            input_status = "Input Feature"
            notes = "Original customer demographic/profile feature"

        schema_rows.append({
            "feature_name": col,
            "dtype": dtype_str,
            "role": role,
            "input_status": input_status,
            "notes": notes
        })

    for col in ["FraudFlag", "FraudType", "Churn"]:
        schema_rows.append({
            "feature_name": col,
            "dtype": "various",
            "role": "Target / Label",
            "input_status": "Leakage (Exclude)",
            "notes": "Target variables to be predicted, excluded from feature matrices to prevent leakage"
        })

    schema_df = pd.DataFrame(schema_rows)
    schema_path = output_dir / "supervised_feature_schema.csv"
    schema_df.to_csv(schema_path, index=False)
    print(f"Saved schema description to {schema_path}")
