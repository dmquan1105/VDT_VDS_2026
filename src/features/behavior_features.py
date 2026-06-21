import pandas as pd


def build_behavior_features(
    device_sessions: pd.DataFrame,
    reference_date
) -> pd.DataFrame:
    df = device_sessions.copy()

    df["SessionDate"] = pd.to_datetime(df["SessionDate"])
    reference_date = pd.to_datetime(reference_date)

    customer_base = df[["CustomerID"]].drop_duplicates()

    # Full observation window

    distinct_ip_count = (
        df.groupby("CustomerID")["IP"]
        .nunique()
        .rename("distinct_ip_count")
        .reset_index()
    )

    distinct_country_count = (
        df.groupby("CustomerID")["CountryCode"]
        .nunique()
        .rename("distinct_country_count")
        .reset_index()
    )

    df["is_datacenter"] = (df["IPType"] == "datacenter").astype(int)
    df["is_vpn_proxy"] = (df["IPType"] == "vpn_proxy").astype(int)

    iptype_features = (
        df.groupby("CustomerID")
        .agg(
            datacenter_ratio=("is_datacenter", "mean"),
            vpn_proxy_ratio=("is_vpn_proxy", "mean"),
        )
        .reset_index()
    )

    iptype_features["non_residential_ratio"] = (
        iptype_features["datacenter_ratio"]
        + iptype_features["vpn_proxy_ratio"]
    )

    home_cell_ratio = (
        df.groupby("CustomerID")["IsHomeCell"]
        .mean()
        .rename("home_cell_ratio")
        .reset_index()
    )

    df["night_session"] = df["SessionHour"].between(0, 5).astype(int)

    night_ratio = (
        df.groupby("CustomerID")["night_session"]
        .mean()
        .rename("night_session_ratio")
        .reset_index()
    )

    active_days = (
        df.groupby("CustomerID")["SessionDate"]
        .nunique()
        .rename("active_days_90d")
        .reset_index()
    )

    session_count = (
        df.groupby("CustomerID")
        .size()
        .rename("total_sessions")
        .reset_index()
    )

    avg_sessions = session_count.merge(
        active_days,
        on="CustomerID",
        how="left"
    )

    avg_sessions["avg_sessions_per_day"] = (
        avg_sessions["total_sessions"]
        / avg_sessions["active_days_90d"]
    )

    avg_sessions = avg_sessions[
        ["CustomerID", "total_sessions", "avg_sessions_per_day"]
    ]

    daily_country_count = (
        df.groupby(["CustomerID", "SessionDate"])["CountryCode"]
        .nunique()
        .reset_index(name="country_count")
    )

    daily_country_count["geo_velocity_day"] = (
        daily_country_count["country_count"] > 1
    ).astype(int)

    geo_velocity = (
        daily_country_count
        .groupby("CustomerID")
        .agg(
            geo_velocity_alerts=("geo_velocity_day", "sum"),
            geo_velocity_flag=("geo_velocity_day", "max"),
        )
        .reset_index()
    )

    first_seen = (
        df.groupby("CustomerID")["SessionDate"]
        .min()
        .rename("first_seen_date")
        .reset_index()
    )

    first_seen["days_since_first_seen"] = (
        reference_date - first_seen["first_seen_date"]
    ).dt.days

    first_seen = first_seen[
        ["CustomerID", "days_since_first_seen"]
    ]

    # Last 30 days window

    cutoff_30d = reference_date - pd.Timedelta(days=30)
    recent_df = df[df["SessionDate"] >= cutoff_30d].copy()

    distinct_ip_30d = (
        recent_df.groupby("CustomerID")["IP"]
        .nunique()
        .rename("distinct_ip_30d")
        .reset_index()
    )

    distinct_country_30d = (
        recent_df.groupby("CustomerID")["CountryCode"]
        .nunique()
        .rename("distinct_country_30d")
        .reset_index()
    )

    recent_df["is_datacenter"] = (
        recent_df["IPType"] == "datacenter"
    ).astype(int)

    recent_df["is_vpn_proxy"] = (
        recent_df["IPType"] == "vpn_proxy"
    ).astype(int)

    iptype_features_30d = (
        recent_df.groupby("CustomerID")
        .agg(
            datacenter_ratio_30d=("is_datacenter", "mean"),
            vpn_proxy_ratio_30d=("is_vpn_proxy", "mean"),
        )
        .reset_index()
    )

    iptype_features_30d["non_residential_ratio_30d"] = (
        iptype_features_30d["datacenter_ratio_30d"]
        + iptype_features_30d["vpn_proxy_ratio_30d"]
    )

    home_cell_ratio_30d = (
        recent_df.groupby("CustomerID")["IsHomeCell"]
        .mean()
        .rename("home_cell_ratio_30d")
        .reset_index()
    )

    recent_df["night_session"] = (
        recent_df["SessionHour"].between(0, 5)
    ).astype(int)

    night_ratio_30d = (
        recent_df.groupby("CustomerID")["night_session"]
        .mean()
        .rename("night_session_ratio_30d")
        .reset_index()
    )

    active_days_30d = (
        recent_df.groupby("CustomerID")["SessionDate"]
        .nunique()
        .rename("active_days_30d")
        .reset_index()
    )

    session_count_30d = (
        recent_df.groupby("CustomerID")
        .size()
        .rename("total_sessions_30d")
        .reset_index()
    )

    avg_sessions_30d = session_count_30d.merge(
        active_days_30d,
        on="CustomerID",
        how="left"
    )

    avg_sessions_30d["avg_sessions_per_day_30d"] = (
        avg_sessions_30d["total_sessions_30d"]
        / avg_sessions_30d["active_days_30d"]
    )

    avg_sessions_30d = avg_sessions_30d[
        ["CustomerID", "total_sessions_30d", "avg_sessions_per_day_30d"]
    ]

    daily_country_count_30d = (
        recent_df.groupby(["CustomerID", "SessionDate"])["CountryCode"]
        .nunique()
        .reset_index(name="country_count")
    )

    daily_country_count_30d["geo_velocity_day"] = (
        daily_country_count_30d["country_count"] > 1
    ).astype(int)

    geo_velocity_30d = (
        daily_country_count_30d
        .groupby("CustomerID")
        .agg(
            geo_velocity_alerts_30d=("geo_velocity_day", "sum"),
            geo_velocity_flag_30d=("geo_velocity_day", "max"),
        )
        .reset_index()
    )

    # Merge all

    features = (
        customer_base
        .merge(distinct_ip_count, on="CustomerID", how="left")
        .merge(distinct_country_count, on="CustomerID", how="left")
        .merge(iptype_features, on="CustomerID", how="left")
        .merge(home_cell_ratio, on="CustomerID", how="left")
        .merge(night_ratio, on="CustomerID", how="left")
        .merge(active_days, on="CustomerID", how="left")
        .merge(avg_sessions, on="CustomerID", how="left")
        .merge(geo_velocity, on="CustomerID", how="left")
        .merge(first_seen, on="CustomerID", how="left")
        .merge(distinct_ip_30d, on="CustomerID", how="left")
        .merge(distinct_country_30d, on="CustomerID", how="left")
        .merge(iptype_features_30d, on="CustomerID", how="left")
        .merge(home_cell_ratio_30d, on="CustomerID", how="left")
        .merge(night_ratio_30d, on="CustomerID", how="left")
        .merge(active_days_30d, on="CustomerID", how="left")
        .merge(avg_sessions_30d, on="CustomerID", how="left")
        .merge(geo_velocity_30d, on="CustomerID", how="left")
    )

    fill_zero_cols = [
        "distinct_ip_count",
        "distinct_country_count",
        "datacenter_ratio",
        "vpn_proxy_ratio",
        "non_residential_ratio",
        "home_cell_ratio",
        "night_session_ratio",
        "active_days_90d",
        "total_sessions",
        "avg_sessions_per_day",
        "geo_velocity_alerts",
        "geo_velocity_flag",
        "distinct_ip_30d",
        "distinct_country_30d",
        "datacenter_ratio_30d",
        "vpn_proxy_ratio_30d",
        "non_residential_ratio_30d",
        "home_cell_ratio_30d",
        "night_session_ratio_30d",
        "active_days_30d",
        "total_sessions_30d",
        "avg_sessions_per_day_30d",
        "geo_velocity_alerts_30d",
        "geo_velocity_flag_30d",
    ]

    for col in fill_zero_cols:
        features[col] = features[col].fillna(0)

    return features