def get_sim_count_per_customer(sim_events):
    return (
        sim_events
        .groupby("CustomerID")["ICCID"]
        .nunique()
        .rename("sim_count")
        .reset_index()
    )