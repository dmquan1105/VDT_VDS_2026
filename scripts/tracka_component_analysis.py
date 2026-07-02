"""Create Track A component submissions and optional sim_swap_ato diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_pipeline import build_feature_matrix
from src.graph.customer_graph import build_customer_graph_edges
from src.models.anomaly import run_isolation_forest
from src.utils.io import load_raw_data
from src.utils.preprocessing import preprocess_unsupervised_holdout

OUTPUT_DIR = PROJECT_ROOT / "outputs"
UNSUP_SUBMIT_DIR = OUTPUT_DIR / "unsup_submit"
DATA_DIR = PROJECT_ROOT / "data"
CONTAMINATION = 0.04

LABEL_CANDIDATES = [
    PROJECT_ROOT / "data" / "dts_holdout_labeled.csv",
    PROJECT_ROOT / "data" / "dts_holdout_answer.csv",
    PROJECT_ROOT / "data" / "dts_holdout_answer_key.csv",
    PROJECT_ROOT / "data" / "holdout_answer_key.csv",
    PROJECT_ROOT / "outputs" / "dts_holdout_labeled.csv",
    PROJECT_ROOT / "outputs" / "holdout_answer_key.csv",
    PROJECT_ROOT / "outputs" / "trackA_holdout_answer_key.csv",
]


COMPONENTS = {
    "iso": "iso_rank_pct",
    "lof": "lof_rank_pct",
    "blend": "anomaly_score",
    "graph_adjusted": "graph_adjusted_anomaly_score",
}

EVAL_COMPONENTS = {
    "iso": "iso_rank_pct",
    "lof": "lof_rank_pct",
    "blend_90_10": "anomaly_score",
    "graph_neighbor": "neighbor_anomaly_rank_pct",
    "graph_adjusted": "graph_adjusted_anomaly_score",
}

LOF_CANDIDATE_COLS = [
    "num_imeis_90d",
    "shared_imei_flag",
    "high_shared_imei_flag",
    "max_customers_per_imei",
    "num_accounts_linked_to_device",
    "is_rooted",
    "rooted_session_ratio",
    "is_emulator",
    "is_generic_or_clone",
    "is_feature_phone",
    "emulator_session_ratio",
    "tac_risk_score",
    "tac_customer_count_max",
    "tac_imei_count_max",
    "tac_customer_per_imei_max",
    "observed_device_days",
    "device_tier_mean",
    "device_tier_min",
    "device_tier_max",
    "low_tier_session_ratio",
    "low_tier_device_flag",
    "device_catalog_missing_ratio",
    "device_catalog_missing_any",
    "tac_grey_clone_flag",
    "phone_number_age_days",
    "sim_swap_count_total",
    "sim_swap_count_90d",
    "sim_swap_count_12m",
    "days_since_last_sim_swap",
    "iccid_count",
    "port_in_flag",
    "recent_sim_change_flag",
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
    "geo_velocity_flag",
    "geo_velocity_alerts",
    "days_since_first_seen",
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
    "geo_velocity_flag_30d",
    "geo_velocity_alerts_30d",
    "kyc_level_ord",
    "has_face_score",
    "has_iddoc_score",
    "face_match_score",
    "id_doc_match_score",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _dts_from_anomaly(score: pd.Series) -> pd.Series:
    """Convert a 0-1 anomaly score into a 0-1000 trust score."""
    return ((1.0 - score.clip(0, 1)) * 1000).round().astype(int)


def _load_component_scores() -> pd.DataFrame:
    """Load and align existing Track A component score outputs."""
    graph_path = OUTPUT_DIR / "anomaly_graph_score.csv"
    base_path = OUTPUT_DIR / "anomaly_scores.csv"
    dts_path = OUTPUT_DIR / "dts_unsup_with_pillars.csv"

    if graph_path.exists():
        scores = _read_csv(graph_path)
    else:
        scores = _read_csv(base_path)

    if "graph_adjusted_anomaly_score" not in scores.columns:
        scores["graph_adjusted_anomaly_score"] = scores["anomaly_score"]

    if dts_path.exists():
        dts = _read_csv(dts_path)
        keep_cols = [
            col
            for col in ["CustomerID", "DTS_unsup", "AnomalyScore"]
            if col in dts.columns and col not in scores.columns
        ]
        if keep_cols:
            scores = scores.merge(dts[["CustomerID", *keep_cols]], on="CustomerID", how="left")

    required = {"CustomerID", *COMPONENTS.values()}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing Track A component columns: {sorted(missing)}")

    return scores


def _write_component_submissions(scores: pd.DataFrame) -> list[Path]:
    """Write one submission CSV per unsupervised component."""
    created: list[Path] = []
    UNSUP_SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

    component_scores = scores[["CustomerID", *COMPONENTS.values()]].copy()
    component_path = UNSUP_SUBMIT_DIR / "tracka_component_scores_holdout.csv"
    component_scores.to_csv(component_path, index=False, encoding="utf-8")
    created.append(component_path)

    for name, score_col in COMPONENTS.items():
        if name == "graph_adjusted" and "DTS_unsup" in scores.columns:
            dts_unsup = scores["DTS_unsup"].round().astype(int)
        else:
            dts_unsup = _dts_from_anomaly(scores[score_col])
        submit = pd.DataFrame(
            {
                "CustomerID": scores["CustomerID"],
                "AnomalyScore": scores[score_col],
                "DTS_unsup": dts_unsup,
            }
        )
        path = UNSUP_SUBMIT_DIR / f"tracka_{name}_submission.csv"
        submit.to_csv(path, index=False, encoding="utf-8")
        created.append(path)

    return created


def _find_holdout_labels() -> tuple[pd.DataFrame | None, Path | None]:
    """Find optional holdout answer key for evaluation."""
    for path in LABEL_CANDIDATES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if {"CustomerID", "FraudFlag", "FraudType"}.issubset(df.columns):
            return df[["CustomerID", "FraudFlag", "FraudType"]].copy(), path
    return None, None


def _recall_at_rate(y_true: np.ndarray, score: np.ndarray, review_rate: float) -> float:
    n_review = int(np.ceil(review_rate * len(y_true)))
    if n_review <= 0 or y_true.sum() == 0:
        return float("nan")
    order = np.argsort(score)[::-1]
    selected = order[:n_review]
    return float(y_true[selected].sum() / y_true.sum())


def _roc_auc_score_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    """Compute ROC-AUC for binary labels using average ranks."""
    y_true = np.asarray(y_true).astype(int)
    score_series = pd.Series(score)
    ranks = score_series.rank(method="average").to_numpy()
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    """Compute average precision for binary labels."""
    y_true = np.asarray(y_true).astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(score)[::-1]
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precision_at_k = tp_cum / ranks
    return float((precision_at_k * y_sorted).sum() / n_pos)


def _preprocess_lof_features(features: pd.DataFrame) -> np.ndarray:
    missing = [col for col in LOF_CANDIDATE_COLS if col not in features.columns]
    non_numeric = [
        col
        for col in LOF_CANDIDATE_COLS
        if col in features.columns and not pd.api.types.is_numeric_dtype(features[col])
    ]
    if missing or non_numeric:
        raise ValueError(
            "LOF candidate schema mismatch: "
            f"missing={missing}, non_numeric={non_numeric}"
        )

    x_raw = features[LOF_CANDIDATE_COLS].copy()
    imputer = SimpleImputer(strategy="median")
    x_imputed = pd.DataFrame(
        imputer.fit_transform(x_raw),
        columns=LOF_CANDIDATE_COLS,
        index=features.index,
    )

    count_like_cols = [
        col
        for col in LOF_CANDIDATE_COLS
        if any(token in col for token in ["count", "distinct", "alerts", "sessions"])
    ]
    long_day_cols = [
        col
        for col in ["days_since_last_sim_swap", "phone_number_age_days"]
        if col in LOF_CANDIDATE_COLS
    ]

    x_transformed = x_imputed.copy()
    for col in long_day_cols:
        x_transformed[col] = x_transformed[col].clip(
            upper=x_transformed[col].quantile(0.99)
        )

    for col in sorted(set(count_like_cols + long_day_cols)):
        x_transformed[col] = x_transformed[col].clip(lower=0)
        x_transformed[col] = np.log1p(x_transformed[col])

    return RobustScaler().fit_transform(x_transformed)


def _neighbor_score_frame(
    customer_ids: pd.Series,
    edges: pd.DataFrame,
    anomaly_scores: pd.Series,
) -> pd.DataFrame:
    score_map = dict(zip(customer_ids, anomaly_scores))
    adjacency: dict[Any, list[Any]] = {customer_id: [] for customer_id in customer_ids}

    if not edges.empty:
        for row in edges[["customer_a", "customer_b"]].itertuples(index=False):
            if row.customer_a in adjacency:
                adjacency[row.customer_a].append(row.customer_b)
            if row.customer_b in adjacency:
                adjacency[row.customer_b].append(row.customer_a)

    rows: list[dict[str, Any]] = []
    for customer_id in customer_ids:
        neighbors = adjacency.get(customer_id, [])
        neighbor_scores = [score_map.get(neighbor, 0.0) for neighbor in neighbors]
        rows.append(
            {
                "CustomerID": customer_id,
                "neighbor_count": len(neighbors),
                "neighbor_anomaly_max": max(neighbor_scores) if neighbor_scores else 0.0,
                "neighbor_anomaly_mean": (
                    float(np.mean(neighbor_scores)) if neighbor_scores else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_train_component_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_data = load_raw_data(DATA_DIR)
    dts_train = raw_data["dts_train"]

    train_features = build_feature_matrix(
        customer_df=dts_train,
        sim_events=raw_data["sim_events"],
        device_sessions=raw_data["device_sessions"],
        kyc_records=raw_data["kyc_records"],
        device_catalog=raw_data["device_catalog"],
    )

    x_iso_processed, _, _ = preprocess_unsupervised_holdout(train_features)
    iso_result = run_isolation_forest(
        x_iso_processed,
        contamination=CONTAMINATION,
        n_estimators=500,
        random_state=42,
    )

    x_lof_processed = _preprocess_lof_features(train_features)
    lof = LocalOutlierFactor(
        n_neighbors=50,
        contamination=CONTAMINATION,
        n_jobs=-1,
    )
    lof.fit_predict(x_lof_processed)

    score_df = pd.DataFrame(
        {
            "CustomerID": train_features["CustomerID"],
            "iso_score": iso_result["score"],
            "lof_score": -lof.negative_outlier_factor_,
        }
    )
    score_df["iso_rank_pct"] = score_df["iso_score"].rank(pct=True)
    score_df["lof_rank_pct"] = score_df["lof_score"].rank(pct=True)
    score_df["anomaly_score"] = (
        0.90 * score_df["iso_rank_pct"]
        + 0.10 * score_df["lof_rank_pct"]
    )

    edges = build_customer_graph_edges(
        device_sessions=raw_data["device_sessions"],
        sim_events=raw_data["sim_events"],
        customer_ids=train_features["CustomerID"],
    )
    neighbor_df = _neighbor_score_frame(
        customer_ids=train_features["CustomerID"],
        edges=edges,
        anomaly_scores=score_df["anomaly_score"],
    )
    score_df = score_df.merge(neighbor_df, on="CustomerID", how="left")
    score_df["neighbor_anomaly_rank_pct"] = score_df["neighbor_anomaly_max"].rank(pct=True)
    score_df["graph_adjusted_anomaly_score"] = (
        0.90 * score_df["anomaly_score"].rank(pct=True)
        + 0.10 * score_df["neighbor_anomaly_rank_pct"]
    )

    labels = dts_train[["CustomerID", "FraudFlag", "FraudType"]].copy()
    return score_df, labels


def _build_missing_label_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, score_col in COMPONENTS.items():
        rows.append(
            {
                "component": name,
                "score_column": score_col,
                "target": "sim_swap_ato",
                "auc": np.nan,
                "pr_auc": np.nan,
                "recall_at_4pct": np.nan,
                "recall_at_5pct": np.nan,
                "sim_swap_ato_count": np.nan,
                "evaluated_rows": 0,
                "status": "missing_holdout_labels",
            }
        )
    return pd.DataFrame(rows)


def _evaluate_sim_swap_ato(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    components: dict[str, str] = COMPONENTS,
    dataset: str = "holdout",
) -> pd.DataFrame:
    """Evaluate each Track A component against sim_swap_ato labels."""
    eval_df = scores.merge(labels, on="CustomerID", how="inner")
    eval_df["is_sim_swap_ato"] = (eval_df["FraudType"] == "sim_swap_ato").astype(int)

    rows: list[dict[str, Any]] = []
    y = eval_df["is_sim_swap_ato"].to_numpy()
    for name, score_col in components.items():
        score = pd.to_numeric(eval_df[score_col], errors="coerce").to_numpy()
        valid = ~np.isnan(score)
        y_valid = y[valid]
        score_valid = score[valid]
        if len(np.unique(y_valid)) < 2:
            auc = np.nan
            pr_auc = np.nan
            status = "not_enough_label_classes"
        else:
            auc = _roc_auc_score_binary(y_valid, score_valid)
            pr_auc = _average_precision_binary(y_valid, score_valid)
            status = "ok"

        rows.append(
            {
                "component": name,
                "score_column": score_col,
                "dataset": dataset,
                "target": "sim_swap_ato",
                "auc": auc,
                "pr_auc": pr_auc,
                "recall_at_4pct": _recall_at_rate(y_valid, score_valid, 0.04),
                "recall_at_5pct": _recall_at_rate(y_valid, score_valid, 0.05),
                "sim_swap_ato_count": int(y_valid.sum()),
                "evaluated_rows": int(valid.sum()),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_blend_weight_sweep(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    eval_df = scores.merge(labels, on="CustomerID", how="inner")
    y = (eval_df["FraudType"] == "sim_swap_ato").astype(int).to_numpy()

    rows: list[dict[str, Any]] = []
    for iso_weight in np.arange(0.0, 1.01, 0.05):
        lof_weight = 1.0 - iso_weight
        score = (
            iso_weight * eval_df["iso_rank_pct"].to_numpy()
            + lof_weight * eval_df["lof_rank_pct"].to_numpy()
        )
        rows.append(
            {
                "dataset": "train",
                "target": "sim_swap_ato",
                "iso_weight": round(float(iso_weight), 2),
                "lof_weight": round(float(lof_weight), 2),
                "auc": _roc_auc_score_binary(y, score),
                "pr_auc": _average_precision_binary(y, score),
                "recall_at_4pct": _recall_at_rate(y, score, 0.04),
                "recall_at_5pct": _recall_at_rate(y, score, 0.05),
                "sim_swap_ato_count": int(y.sum()),
                "evaluated_rows": int(len(eval_df)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["recall_at_4pct", "pr_auc", "auc"],
        ascending=False,
    )


def main() -> None:
    scores = _load_component_scores()
    created = _write_component_submissions(scores)

    labels, label_path = _find_holdout_labels()
    if labels is None:
        metrics = _build_missing_label_metrics()
        label_source = None
        warning = (
            "Holdout labels with CustomerID, FraudFlag, FraudType were not found; "
            "sim_swap_ato AUC/recall metrics are marked missing."
        )
        print(f"WARNING: {warning}")
    else:
        metrics = _evaluate_sim_swap_ato(scores, labels)
        label_source = str(label_path.relative_to(PROJECT_ROOT)) if label_path else None
        warning = None

    metrics_path = UNSUP_SUBMIT_DIR / "tracka_component_sim_swap_ato_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    created.append(metrics_path)

    print("Building train Track A component scores for labeled sim_swap_ato diagnostics...")
    train_scores, train_labels = _build_train_component_scores()

    train_scores_path = UNSUP_SUBMIT_DIR / "tracka_train_component_scores.csv"
    train_scores.to_csv(train_scores_path, index=False, encoding="utf-8")
    created.append(train_scores_path)

    train_metrics = _evaluate_sim_swap_ato(
        train_scores,
        train_labels,
        components=EVAL_COMPONENTS,
        dataset="train",
    )
    train_metrics_path = UNSUP_SUBMIT_DIR / "tracka_train_component_sim_swap_ato_metrics.csv"
    train_metrics.to_csv(train_metrics_path, index=False, encoding="utf-8")
    created.append(train_metrics_path)

    blend_sweep = _evaluate_blend_weight_sweep(train_scores, train_labels)
    blend_sweep_path = UNSUP_SUBMIT_DIR / "tracka_train_blend_weight_sweep.csv"
    blend_sweep.to_csv(blend_sweep_path, index=False, encoding="utf-8")
    created.append(blend_sweep_path)

    manifest = {
        "output_dir": str(UNSUP_SUBMIT_DIR.relative_to(PROJECT_ROOT)),
        "label_source": label_source,
        "warning": warning,
        "component_score_columns": COMPONENTS,
        "train_eval_component_score_columns": EVAL_COMPONENTS,
        "created_files": [str(path.relative_to(PROJECT_ROOT)) for path in created],
        "label_candidates_checked": [str(path.relative_to(PROJECT_ROOT)) for path in LABEL_CANDIDATES],
    }
    manifest_path = UNSUP_SUBMIT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    created.append(manifest_path)

    print("Created/updated files:")
    for path in created:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
