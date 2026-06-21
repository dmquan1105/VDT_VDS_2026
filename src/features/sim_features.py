import pandas as pd

def build_sim_features(sim_events: pd.DataFrame, reference_date) -> pd.DataFrame:
    # reference_date = max(
    #     pd.to_datetime(sim_events["EventDate"]).max(),
    #     pd.to_datetime(device_sessions["SessionDate"]).max()
    # )
    
    df = sim_events.copy()
    
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    reference_date = pd.to_datetime(reference_date)
    
    swap_df = df[
        df["EventType"] == "sim_swap"
    ].copy()

    cutoff_90d = reference_date - pd.Timedelta(days=90)
    cutoff_12m = reference_date - pd.Timedelta(days=365)
    
    customer_base = df[["CustomerID"]].drop_duplicates()
    
    # Phone number age
    activation = (
        df[df["EventType"] == "number_activation"]
        .groupby("CustomerID")["EventDate"]
        .min()
        .rename("activation_date")
        .reset_index()
    )
    
    activation["phone_number_age_days"] = (
        reference_date - activation["activation_date"]
    ).dt.days
    
    # SIM swap count (90D, 12M, total)
    sim_swap_count_total = (
        df[df["EventType"] == "sim_swap"]
        .groupby("CustomerID")
        .size()
        .rename("sim_swap_count_total")
        .reset_index()
    )
    
    sim_swap_count_90d = (
        swap_df[
            swap_df["EventDate"] >= cutoff_90d
        ]
        .groupby("CustomerID")
        .size()
        .rename("sim_swap_count_90d")
        .reset_index()
    )
    
    sim_swap_count_12m = (
        swap_df[
            swap_df["EventDate"] >= cutoff_12m
        ]
        .groupby("CustomerID")
        .size()
        .rename("sim_swap_count_12m")
        .reset_index()
    )
    
    
    
    # Days since last SIM swap
    last_swap = (
        df[df["EventType"] == "sim_swap"]
        .groupby("CustomerID")["EventDate"]
        .max()
        .rename("last_sim_swap_date")
        .reset_index()
    )
    
    last_swap["days_since_last_sim_swap"] = (
        reference_date - last_swap["last_sim_swap_date"]
    ).dt.days
    
    # iccid count
    iccid_count = (
        df.groupby("CustomerID")["ICCID"]
        .nunique()
        .rename("iccid_count")
        .reset_index()
    )
    
    # Port-in flag
    port_in_flag = (
        df.assign(port_in_flag=(df["EventType"] == "port_in").astype(int))
        .groupby("CustomerID")["port_in_flag"]
        .max()
        .reset_index()
    )
    
    features = (
        customer_base
        .merge(activation[["CustomerID", "phone_number_age_days"]], on="CustomerID", how="left")
        .merge(sim_swap_count_total, on="CustomerID", how="left")
        .merge(sim_swap_count_90d, on="CustomerID", how="left")
        .merge(sim_swap_count_12m, on="CustomerID", how="left")
        .merge(last_swap[["CustomerID", "days_since_last_sim_swap"]], on="CustomerID", how="left")
        .merge(iccid_count, on="CustomerID", how="left")
        .merge(port_in_flag, on="CustomerID", how="left")
    )
    
    swap_cols = ["sim_swap_count_total", "sim_swap_count_90d", "sim_swap_count_12m"]
    for col in swap_cols:
        features[col] = features[col].fillna(0).astype(int)
        
    features["port_in_flag"] = features["port_in_flag"].fillna(0).astype(int)
    features["iccid_count"] = features["iccid_count"].fillna(0).astype(int)

    features["days_since_last_sim_swap"] = (
        features["days_since_last_sim_swap"]
        .fillna(9999)
        .astype(int)
    )
    
    features["recent_sim_change_flag"] = (
        features["days_since_last_sim_swap"] <= 30
    ).astype(int)

    return features