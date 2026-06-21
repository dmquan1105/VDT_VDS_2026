from itertools import combinations

import networkx as nx
import pandas as pd

def _normalize_customer_pair(a, b):
    # Giữ thứ tự ổn định giữa các khách hàng
    return (a, b) if a < b else (b, a)

def _make_pair_edges_from_entity(
    df: pd.DataFrame,
    entity_col: str,
    relation_type: str,
    edge_weight: float,
    max_group_size: int = 30,
    min_sessions_per_customer: int = 1,
) -> pd.DataFrame:
    """
    Dựng cạnh customer-customer khi nhiều customers dùng cùng 1 entity (Ví dụ như IMEI, ...)

    Ví dụ:
        Customer A -- IMEI X
        Customer B -- IMEI X

    Thành:
        A --shared_imei--> B
    """
    
    if df.empty:
        return pd.DataFrame(
            columns=[
                "customer_a",
                "customer_b",
                "relation_type",
                "entity_type",
                "entity_value",
                "edge_weight",
                "entity_customer_count",
            ]
        )

    x = (
        df[["CustomerID", entity_col]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    customer_entity_counts = (
        df[["CustomerID", entity_col]]
        .dropna()
        .groupby([entity_col, "CustomerID"])
        .size()
        .rename("customer_entity_sessions")
        .reset_index()
    )

    if min_sessions_per_customer > 1:
        customer_entity_counts = customer_entity_counts[
            customer_entity_counts["customer_entity_sessions"]
            >= min_sessions_per_customer
        ]

    rows = []

    for entity_value, group in customer_entity_counts.groupby(entity_col):
        customers = sorted(group["CustomerID"].unique())
        n_customers = len(customers)

        if n_customers < 2:
            continue

        if n_customers > max_group_size:
            continue

        for a, b in combinations(customers, 2):
            customer_a, customer_b = _normalize_customer_pair(a, b)

            rows.append(
                {
                    "customer_a": customer_a,
                    "customer_b": customer_b,
                    "relation_type": relation_type,
                    "entity_type": entity_col,
                    "entity_value": str(entity_value),
                    "edge_weight": edge_weight,
                    "entity_customer_count": n_customers,
                }
            )

    return pd.DataFrame(rows)

def build_customer_graph_edges(
    device_sessions: pd.DataFrame,
    sim_events: pd.DataFrame,
    customer_ids=None,
) -> pd.DataFrame:
    """
    Dựng đồ thị customer-customer từ các entities dùng chung.

    Output là danh sách các cạnh:
        customer_a, customer_b, relation_type, entity_type, entity_value, edge_weight
    """
    sessions = device_sessions.copy()
    sims = sim_events.copy()

    if customer_ids is not None:
        customer_ids = set(customer_ids)
        sessions = sessions[sessions["CustomerID"].isin(customer_ids)].copy()
        sims = sims[sims["CustomerID"].isin(customer_ids)].copy()

    sessions = sessions.drop_duplicates()
    sims = sims.drop_duplicates()

    sessions["IPType_norm"] = (
        sessions["IPType"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    edge_frames = []

    # Dùng chung thiết bị vật lý (Chung IMEI)
    imei_edges = _make_pair_edges_from_entity(
        df=sessions,
        entity_col="IMEI",
        relation_type="shared_imei",
        edge_weight=5.0,
        max_group_size=20,
        min_sessions_per_customer=1,
    )
    edge_frames.append(imei_edges)

    # Dùng chung SIM ICCID 
    iccid_edges = _make_pair_edges_from_entity(
        df=sims,
        entity_col="ICCID",
        relation_type="shared_iccid",
        edge_weight=6.0,
        max_group_size=10,
        min_sessions_per_customer=1,
    )
    edge_frames.append(iccid_edges)

    # 3. Risky IP: datacenter / VPN / proxy.
    risky_ip_sessions = sessions[
        sessions["IPType_norm"].isin(["datacenter", "vpn_proxy", "vpn", "proxy"])
    ].copy()

    risky_ip_edges = _make_pair_edges_from_entity(
        df=risky_ip_sessions,
        entity_col="IP",
        relation_type="shared_risky_ip",
        edge_weight=3.0,
        max_group_size=50,
        min_sessions_per_customer=1,
    )
    edge_frames.append(risky_ip_edges)

    # Dùng chung IP đáng ngờ trong cùng ngày
    if not risky_ip_sessions.empty:
        risky_ip_sessions["IP_Date"] = (
            risky_ip_sessions["IP"].astype(str)
            + "|"
            + risky_ip_sessions["SessionDate"].astype(str)
        )

        risky_ip_same_day_edges = _make_pair_edges_from_entity(
            df=risky_ip_sessions,
            entity_col="IP_Date",
            relation_type="shared_risky_ip_same_day",
            edge_weight=4.0,
            max_group_size=30,
            min_sessions_per_customer=1,
        )
        edge_frames.append(risky_ip_same_day_edges)

    # Quan hệ yếu, chia sẻ giữa WIfi hộ dân
    residential_sessions = sessions[
        sessions["IPType_norm"].eq("residential_wifi")
    ].copy()

    residential_edges = _make_pair_edges_from_entity(
        df=residential_sessions,
        entity_col="IP",
        relation_type="shared_residential_wifi_repeated",
        edge_weight=1.0,
        max_group_size=10,
        min_sessions_per_customer=2,
    )
    edge_frames.append(residential_edges)

    raw_edges = pd.concat(edge_frames, ignore_index=True)

    if raw_edges.empty:
        return pd.DataFrame(
            columns=[
                "customer_a",
                "customer_b",
                "total_edge_weight",
                "relation_count",
                "evidence_count",
                "relations",
                "evidence_examples",
            ]
        )

    # Tổng hợp các rows giữa các cặp customer
    edge_df = (
        raw_edges.groupby(["customer_a", "customer_b"])
        .agg(
            total_edge_weight=("edge_weight", "sum"),
            max_edge_weight=("edge_weight", "max"),
            relation_count=("relation_type", "nunique"),
            evidence_count=("relation_type", "size"),
            max_entity_customer_count=("entity_customer_count", "max"),
            relations=(
                "relation_type",
                lambda x: "|".join(sorted(set(x))),
            ),
            evidence_examples=(
                "entity_value",
                lambda x: "|".join(list(x.astype(str).head(5))),
            ),
        )
        .reset_index()
    )

    return edge_df.sort_values(
        ["total_edge_weight", "evidence_count"],
        ascending=False,
    ).reset_index(drop=True)
    
def build_networkx_customer_graph(edge_df: pd.DataFrame) -> nx.Graph:
    """
    Dựng đồ thị NetworkX từ danh sách cạnh customer-customer 
    """
    G = nx.Graph()

    for row in edge_df.itertuples(index=False):
        G.add_edge(
            int(row.customer_a),
            int(row.customer_b),
            weight=float(row.total_edge_weight),
            max_edge_weight=float(row.max_edge_weight),
            relation_count=int(row.relation_count),
            evidence_count=int(row.evidence_count),
            relations=row.relations,
            evidence_examples=row.evidence_examples,
        )

    return G

def get_customer_neighbors(edge_df: pd.DataFrame, customer_id: int) -> pd.DataFrame:
    """
    # Trả về đồ thị hàng xóm trực tiếp của 1 customer
    """
    mask = (
        (edge_df["customer_a"] == customer_id)
        | (edge_df["customer_b"] == customer_id)
    )

    neighbors = edge_df[mask].copy()

    if neighbors.empty:
        return neighbors

    neighbors["neighbor_customer"] = neighbors.apply(
        lambda row: row["customer_b"]
        if row["customer_a"] == customer_id
        else row["customer_a"],
        axis=1,
    )

    return neighbors.sort_values(
        ["total_edge_weight", "evidence_count"],
        ascending=False,
    )
    
def get_ego_graph(
    G: nx.Graph,
    customer_id: int,
    radius: int = 1,
) -> nx.Graph:
    # Trả về đồ thị con gồm node trung tâm (ego node) và tất cả các hàng xóm quanh nó trong bán kính radius
    if customer_id not in G:
        return nx.Graph()

    return nx.ego_graph(
        G,
        customer_id,
        radius=radius,
    )