import pandas as pd

def build_device_features(
    device_sessions: pd.DataFrame,
    device_catalog: pd.DataFrame,
    drop_duplicates: bool = True
) -> pd.DataFrame:
    df = device_sessions.copy()
    catalog = device_catalog.copy()
    
    if drop_duplicates:
        df = df.drop_duplicates()
    
    df["SessionDate"] = pd.to_datetime(df["SessionDate"])
    
    # Extract TAC from IMEI
    df["TAC"] = df["IMEI"].astype(str).str[:8]
    catalog["TAC"] = catalog["TAC"].astype(str)
    
    # Join device catalog
    df = df.merge(
        catalog,
        on="TAC",
        how="left"
    )
    
    df["device_catalog_missing_flag"] = (
        df["DeviceTier"].isna()
        | df["DeviceBrand"].isna()
        | df["DeviceModel"].isna()
        | df["DeviceOS"].isna()
    ).astype(int)
    
    customer_base = df[["CustomerID"]].drop_duplicates()
    
    # Number of IMEI per customer
    num_imeis = (
        df.groupby("CustomerID")["IMEI"]
        .nunique()
        .rename("num_imeis_90d")
        .reset_index()
    )
    
    # Accounts linked to each IMEI
    imei_customer_count = (
        df.groupby("IMEI")["CustomerID"]
        .nunique()
        .rename("accounts_per_IMEI")
        .reset_index()
    )
    
    df = df.merge(
        imei_customer_count,
        on="IMEI",
        how="left"
    )
    
    # Max customers per IMEI for each customer
    max_customers_per_imei = (
        df.groupby("CustomerID")["accounts_per_IMEI"]
        .max()
        .rename("max_customers_per_imei")
        .reset_index()
    )
    
    num_accounts_linked_to_device = (
        max_customers_per_imei
        .rename(
            columns={
                "max_customers_per_imei": "num_accounts_linked_to_device"
            }
        )
    )
    
    # Shared IMEI flag
    shared_imei_flag = max_customers_per_imei.copy()
    shared_imei_flag["shared_imei_flag"] = (
        shared_imei_flag["max_customers_per_imei"] > 1
    ).astype(int)
    shared_imei_flag = shared_imei_flag[
        ["CustomerID", "shared_imei_flag"]
    ]

    # Stronger shared-device flag
    # Ngưỡng 4 giải thích được từ EDA:
    # fraud rate bắt đầu tăng rất mạnh khi max_customers_per_imei >= 4.
    high_shared_imei_flag = max_customers_per_imei.copy()
    high_shared_imei_flag["high_shared_imei_flag"] = (
        high_shared_imei_flag["max_customers_per_imei"] >= 4
    ).astype(int)
    high_shared_imei_flag = high_shared_imei_flag[
        ["CustomerID", "high_shared_imei_flag"]
    ]
    
    shared_imei_flag = shared_imei_flag[
        ["CustomerID", "shared_imei_flag"]
    ]
    
    # Root device features
    rooted_features = (
        df.groupby("CustomerID")
        .agg(
            rooted_session_ratio=("RootedFlag", "mean"),
            is_rooted=("RootedFlag", "max"),
        )
        .reset_index()
    )
    
    # Observed device days
    observed_device_days = (
        df.groupby("CustomerID")["SessionDate"]
        .nunique()
        .rename("observed_device_days")
        .reset_index()
    )
    
    # Device tier >= 3 được coi là low-tier risk 
    df["low_tier_session_flag"] = (
        df["DeviceTier"].fillna(0) >= 3
    ).astype(int)
    
    device_tier_features = (
        df.groupby("CustomerID")
        .agg(
            device_tier_mean=("DeviceTier", "mean"),
            device_tier_min=("DeviceTier", "min"),
            device_tier_max=("DeviceTier", "max"),
            low_tier_session_ratio=("low_tier_session_flag", "mean"),
            low_tier_device_flag=("low_tier_session_flag", "max"),
        )
        .reset_index()
    )
    
    device_catalog_missing_features = (
        df.groupby("CustomerID")
        .agg(
            device_catalog_missing_ratio=("device_catalog_missing_flag", "mean"),
            device_catalog_missing_any=("device_catalog_missing_flag", "max"),
        )
        .reset_index()
    )
    
    # Emulator / Generic / Clone qua TAC catalog
    brand = df["DeviceBrand"].fillna("").astype(str).str.lower()
    model = df["DeviceModel"].fillna("").astype(str).str.lower()
    os_name = df["DeviceOS"].fillna("").astype(str).str.lower()
    
    suspicious_catalog_pattern = (
        brand.str.contains("generic", regex=True)
        | model.str.contains("clone|emulator|x86", regex=True)
        | os_name.str.contains("feature|kaios", regex=True)
    )

    df["is_emulator_session"] = suspicious_catalog_pattern.astype(int)
    
    emulator_features = (
        df.groupby("CustomerID")
        .agg(
            is_emulator=("is_emulator_session", "max"),
            emulator_session_ratio=("is_emulator_session", "mean"),
        )
        .reset_index()
    )
    
    # TAC features
    tac_stats = (
        df.groupby("TAC")
        .agg(
            tac_imei_count=("IMEI", "nunique"),
            tac_customer_count=("CustomerID", "nunique"),
        )
        .reset_index()
    )

    tac_stats["tac_customer_per_imei"] = (
        tac_stats["tac_customer_count"]
        / tac_stats["tac_imei_count"].replace(0, pd.NA)
    )
    
    df = df.merge(
        tac_stats,
        on="TAC",
        how="left"
    )

    tac_features = (
        df.groupby("CustomerID")
        .agg(
            tac_customer_count_max=("tac_customer_count", "max"),
            tac_imei_count_max=("tac_imei_count", "max"),
            tac_customer_per_imei_max=("tac_customer_per_imei", "max"),
        )
        .reset_index()
    )
    
    # TAC grey/clone proxy
    # - Catalog nói là emulator/generic/clone
    # - hoặc low-tier + IMEI bị dùng bởi >=4 account
    # - hoặc rooted + IMEI bị dùng bởi >=4 account
    
    
    features = (
        customer_base
        .merge(num_imeis, on="CustomerID", how="left")
        .merge(max_customers_per_imei, on="CustomerID", how="left")
        .merge(num_accounts_linked_to_device, on="CustomerID", how="left")
        .merge(shared_imei_flag, on="CustomerID", how="left")
        .merge(high_shared_imei_flag, on="CustomerID", how="left")
        .merge(rooted_features, on="CustomerID", how="left")
        .merge(observed_device_days, on="CustomerID", how="left")
        .merge(device_tier_features, on="CustomerID", how="left")
        .merge(device_catalog_missing_features, on="CustomerID", how="left")
        .merge(emulator_features, on="CustomerID", how="left")
        .merge(tac_features, on="CustomerID", how="left")
    )

    fill_zero_cols = [
        "num_imeis_90d",
        "max_customers_per_imei",
        "num_accounts_linked_to_device",
        "shared_imei_flag",
        "high_shared_imei_flag",
        "rooted_session_ratio",
        "is_rooted",
        "observed_device_days",
        "device_tier_mean",
        "device_tier_min",
        "device_tier_max",
        "low_tier_session_ratio",
        "low_tier_device_flag",
        "is_emulator",
        "emulator_session_ratio",
        "tac_customer_count_max",
        "tac_imei_count_max",
        "tac_customer_per_imei_max",
        "device_catalog_missing_ratio",
        "device_catalog_missing_any",
    ]

    for col in fill_zero_cols:
        features[col] = features[col].fillna(0)
        
    features["tac_grey_clone_flag"] = (
        (features["is_emulator"] == 1)
        | (
            (features["low_tier_device_flag"] == 1)
            & (features["high_shared_imei_flag"] == 1)
        )
        | (
            (features["is_rooted"] == 1)
            & (features["high_shared_imei_flag"] == 1)
        )
    ).astype(int)

    features["tac_risk_score"] = (
        2 * features["is_emulator"]
        + 1 * features["low_tier_device_flag"]
        + 1 * features["high_shared_imei_flag"]
        + 1 * features["is_rooted"]
    )

    return features