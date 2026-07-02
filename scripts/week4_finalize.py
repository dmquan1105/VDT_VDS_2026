"""Finalize week 4 artifacts for Device Trust Score."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.business.decision_matrix import assign_decision_actions, build_decision_matrix
from src.evaluation.fairness_segments import build_segment_report
from src.evaluation.psi import build_psi_report


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "supervised"
PSI_DETAIL_DIR = OUTPUT_DIR / "psi_details"
WARNINGS: list[str] = []

FEATURE_CANDIDATES = [
    "kyc_level_ord",
    "identity_confidence_score",
    "phone_number_age_days",
    "sim_swap_count_90d",
    "days_since_last_sim_swap",
    "iccid_count",
    "port_in_flag",
    "recent_sim_change_flag",
    "num_imeis_90d",
    "max_customers_per_imei",
    "shared_imei_flag",
    "rooted_session_ratio",
    "rooted_flag_any",
    "is_rooted",
    "device_tier",
    "DeviceTier",
    "device_tier_min",
    "device_tier_mean",
    "device_age_days",
    "observed_device_days",
    "distinct_ip_count",
    "distinct_country_count",
    "datacenter_ratio",
    "vpn_proxy_ratio",
    "non_residential_ratio",
    "home_cell_ratio",
    "night_session_ratio",
    "active_days_90d",
    "avg_sessions_per_day",
    "geo_velocity_flag",
    "MonthlyMinutes",
    "RoamingCalls",
    "NonUSTravel",
]


def _warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"WARNING: {message}")


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
    except Exception as exc:
        _warn(f"Cannot read {path.relative_to(PROJECT_ROOT)}: {exc}")
        return None
    _warn(f"Unsupported table format: {path.relative_to(PROJECT_ROOT)}")
    return None


def _write_csv(df: pd.DataFrame, path: Path, created: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    created.append(path)


def _write_json(data: dict[str, Any], path: Path, created: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    created.append(path)


def _select_final_model() -> str:
    return "xgb_tuned"


def _model_aliases(model_name: str) -> list[str]:
    aliases = [model_name]
    if model_name.endswith("_tuned"):
        aliases.append(model_name.replace("_tuned", ""))
    aliases.extend(["xgb_tuned", "xgb"] if model_name.startswith("xgb") else [])
    return list(dict.fromkeys(aliases))


def _find_final_scored(model_name: str) -> pd.DataFrame:
    candidates = [
        OUTPUT_DIR / "holdout_submission_final.csv",
        OUTPUT_DIR / "holdout_submission_v2.csv",
        OUTPUT_DIR / "supervised_holdout_submission.csv",
        OUTPUT_DIR / "holdout_scored.csv",
    ]
    for path in candidates:
        if path.exists():
            df = _read_table(path)
            if df is not None and {"CustomerID", "P_fraud", "DTS"}.issubset(df.columns):
                return df[["CustomerID", "P_fraud", "DTS"]].copy()

    submit_dir = OUTPUT_DIR / "submit"
    for alias in _model_aliases(model_name):
        path = submit_dir / f"{alias}.csv"
        if path.exists():
            df = _read_table(path)
            if df is not None and {"CustomerID", "P_fraud", "DTS"}.issubset(df.columns):
                return df[["CustomerID", "P_fraud", "DTS"]].copy()

    calibrated_path = OUTPUT_DIR / "supervised_holdout_calibrated_scores.csv"
    if calibrated_path.exists():
        df = _read_table(calibrated_path)
        if df is not None:
            aliases = set(_model_aliases(model_name))
            if "model_name" in df.columns:
                df = df[df["model_name"].isin(aliases)]
            if {"CustomerID", "calibrated_p_fraud", "pdo_score"}.issubset(df.columns):
                return df.rename(columns={"calibrated_p_fraud": "P_fraud", "pdo_score": "DTS"})[
                    ["CustomerID", "P_fraud", "DTS"]
                ].copy()

    for path in sorted(OUTPUT_DIR.rglob("*.csv")):
        df = _read_table(path)
        if df is not None and {"CustomerID", "P_fraud", "DTS"}.issubset(df.columns):
            _warn(f"Using discovered scored file {path.relative_to(PROJECT_ROOT)}.")
            return df[["CustomerID", "P_fraud", "DTS"]].copy()

    raise FileNotFoundError(
        "No final holdout scored file found. Run notebooks/supervised.ipynb first to create CustomerID, P_fraud, DTS."
    )


def _find_train_score_df(final_model: str) -> pd.DataFrame | None:
    candidate_names = [
        "train_oof_scores.csv",
        "oof_scores.csv",
        "supervised_train_oof_scores.csv",
        "train_scored.csv",
    ]
    for name in candidate_names:
        for path in [OUTPUT_DIR / name, PROJECT_ROOT / "outputs" / name]:
            if path.exists():
                df = _read_table(path)
                if df is not None and "CustomerID" in df.columns:
                    if "model_name" in df.columns:
                        df = df[df["model_name"].isin(_model_aliases(final_model))]
                    if "P_fraud" not in df.columns:
                        for col in ["y_prob_calibrated", "calibrated_p_fraud", "pred_proba", "oof_pred"]:
                            if col in df.columns:
                                df = df.rename(columns={col: "P_fraud"})
                                break
                    if "DTS" not in df.columns and "pdo_score" in df.columns:
                        df = df.rename(columns={"pdo_score": "DTS"})
                    if {"P_fraud", "DTS"}.issubset(df.columns):
                        return df[["CustomerID", "P_fraud", "DTS"]].copy()
    return None


def _find_feature_table(kind: str) -> pd.DataFrame | None:
    names = [
        f"{kind}_features_supervised.parquet",
        f"{kind}_features.parquet",
        f"{kind}_features_supervised.csv",
        f"{kind}_features.csv",
    ]
    for directory in [OUTPUT_DIR, PROJECT_ROOT / "outputs"]:
        for name in names:
            path = directory / name
            if path.exists():
                df = _read_table(path)
                if df is not None:
                    return df
    fallback = PROJECT_ROOT / "data" / ("dts_train.csv" if kind == "train" else "dts_holdout.csv")
    if fallback.exists():
        _warn(f"Using {fallback.relative_to(PROJECT_ROOT)} as fallback {kind} feature table.")
        return _read_table(fallback)
    return None


def _feature_columns(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> list[str]:
    shap_path = OUTPUT_DIR / "supervised_global_shap_importance.csv"
    cols: list[str] = []
    if shap_path.exists():
        shap_df = _read_table(shap_path)
        if shap_df is not None and "feature" in shap_df.columns:
            cols.extend(shap_df["feature"].dropna().astype(str).head(20).tolist())
    cols.extend(FEATURE_CANDIDATES)
    common = [c for c in dict.fromkeys(cols) if c in train_df.columns and c in holdout_df.columns]
    return common


def _create_missing_score_psi(created: list[Path]) -> None:
    path = OUTPUT_DIR / "score_psi_report.csv"
    df = pd.DataFrame(
        [
            {
                "variable": "P_fraud",
                "psi": "",
                "status": "missing_train_score_df",
                "type": "score",
                "missing_train": "",
                "missing_holdout": "",
            },
            {
                "variable": "DTS",
                "psi": "",
                "status": "missing_train_score_df",
                "type": "score",
                "missing_train": "",
                "missing_holdout": "",
            },
        ]
    )
    _write_csv(df, path, created)


def _model_card(final_model: str, thresholds: dict[str, Any], created: list[Path]) -> None:
    metrics_note = (
        "Theo summary hiện có: xgb_tuned AUC khoảng 0.954, PR-AUC khoảng 0.797, "
        "Brier khoảng 0.0112, Precision@5% khoảng 0.508; stacking_tuned nhỉnh hơn rất nhỏ "
        "nhưng phức tạp hơn để vận hành."
    )
    text = f"""# Final Model Card - Device Trust Score

## Model name

`{final_model}`.

## Intended use

- Chấm `P_fraud` đã calibration và `DTS` 0-1000 cho tập holdout/production.
- Hỗ trợ KYC, fraud review và credit input theo decision matrix.
- Fraud review bị giới hạn theo operating point top 5% risk.

## Not intended use

- Không dùng DTS để auto reject tín dụng một mình.
- Không dùng score thay thế điều tra fraud, KYC policy, hoặc credit score truyền thống.
- Không dùng cho population khác nếu chưa kiểm tra drift, calibration và proxy bias.

## Data

Train có 51,047 khách hàng có nhãn; holdout có 20,000 khách hàng. Event tables SIM, session, KYC và catalog được tổng hợp về cấp `CustomerID`.

## Validation and metrics

{metrics_note}

Metric chi tiết cần lấy lại từ notebook/output supervised nếu muốn audit từng fold.

## Calibration

Pipeline dùng xác suất fraud đã calibration trước khi đổi sang PDO/DTS. `P_fraud` cao tương ứng `DTS` thấp.

## Decision thresholds

- `kyc_fast_cutoff`: {thresholds.get("kyc_fast_cutoff"):.8f}
- `kyc_reject_cutoff`: {thresholds.get("kyc_reject_cutoff"):.8f}
- `fraud_stepup_cutoff_p`: {thresholds.get("fraud_stepup_cutoff_p"):.8f}
- `fraud_review_cutoff_p`: {thresholds.get("fraud_review_cutoff_p"):.8f}
- `fraud_decline_cutoff_p`: {thresholds.get("fraud_decline_cutoff_p"):.8f}
- `credit_low_cutoff`: {thresholds.get("credit_low_cutoff"):.8f}
- `credit_high_cutoff`: {thresholds.get("credit_high_cutoff"):.8f}
- `expected_review_rate`: {thresholds.get("expected_review_rate"):.2%}
- `manual_review_rate`: {thresholds.get("manual_review_rate"):.2%}
- `decline_or_block_rate`: {thresholds.get("decline_or_block_rate"):.2%}
- `manual_review_or_decline_rate`: {thresholds.get("manual_review_or_decline_rate"):.2%}

## Reason for model choice

`xgb_tuned` được chọn mặc định vì hiệu năng gần tương đương stacking nhưng đơn giản hơn để deploy, kiểm thử và giải thích bằng SHAP trực tiếp.

## Limitations

- Dữ liệu mang tính synthetic/competition nên cần kiểm chứng ngoài thực tế.
- Label fraud có thể đến trễ, nhất là mule/subscription fraud.
- Một số feature có thể là proxy cho hành vi hợp pháp như roaming, port-in hoặc thiết bị giá rẻ.

## Monitoring

Theo dõi PSI feature/score, calibration drift, fraud rate theo DTS bucket, review precision, recall theo FraudType khi label đến, và các feature dễ bị game như SIM swap, shared device, VPN/datacenter, KYC missing.
"""
    path = OUTPUT_DIR / "final_model_card.md"
    path.write_text(text, encoding="utf-8")
    created.append(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PSI_DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    final_model = _select_final_model()
    scored = _find_final_scored(final_model)
    scored["P_fraud"] = pd.to_numeric(scored["P_fraud"], errors="coerce")
    scored["DTS"] = pd.to_numeric(scored["DTS"], errors="coerce")
    scored = scored.dropna(subset=["CustomerID", "P_fraud", "DTS"]).copy()

    final_submission_path = OUTPUT_DIR / "holdout_submission_final.csv"
    _write_csv(scored[["CustomerID", "P_fraud", "DTS"]], final_submission_path, created)

    decisions, thresholds = assign_decision_actions(scored, review_rate=0.05, extreme_rate=0.005)
    thresholds["final_model"] = final_model
    decision_matrix = build_decision_matrix(thresholds)
    _write_csv(decision_matrix, OUTPUT_DIR / "decision_matrix.csv", created)
    _write_csv(decisions, OUTPUT_DIR / "holdout_decision_actions.csv", created)
    _write_json(thresholds, OUTPUT_DIR / "decision_thresholds.json", created)

    train_score = _find_train_score_df(final_model)
    if train_score is not None:
        score_report, score_details = build_psi_report(train_score, decisions, ["P_fraud", "DTS"])
        _write_csv(score_report, OUTPUT_DIR / "score_psi_report.csv", created)
        for variable, detail in score_details.items():
            _write_csv(detail, PSI_DETAIL_DIR / f"score_{variable}_psi_detail.csv", created)
    else:
        _warn("Train OOF score file not found; score PSI is marked missing.")
        _create_missing_score_psi(created)

    train_features = _find_feature_table("train")
    holdout_features = _find_feature_table("holdout")
    if train_features is not None and holdout_features is not None:
        feature_cols = _feature_columns(train_features, holdout_features)
        if feature_cols:
            feature_report, feature_details = build_psi_report(train_features, holdout_features, feature_cols)
            feature_report = feature_report.sort_values("psi", ascending=False, na_position="last")
            _write_csv(feature_report, OUTPUT_DIR / "feature_psi_report.csv", created)
            for variable, detail in feature_details.items():
                safe_name = variable.replace("/", "_").replace(" ", "_")
                _write_csv(detail, PSI_DETAIL_DIR / f"feature_{safe_name}_psi_detail.csv", created)
        else:
            _warn("No overlapping selected feature columns found for feature PSI.")
    else:
        _warn("Feature PSI skipped because train or holdout feature table is missing.")

    label_df = _read_table(PROJECT_ROOT / "data" / "dts_train.csv")
    _write_csv(
        build_segment_report(decisions, feature_df=holdout_features),
        OUTPUT_DIR / "fairness_segment_report.csv",
        created,
    )
    if train_features is not None and train_score is not None and label_df is not None:
        train_decisions, _ = assign_decision_actions(train_score, review_rate=0.05, extreme_rate=0.005)
        _write_csv(
            build_segment_report(train_decisions, feature_df=train_features, label_df=label_df),
            OUTPUT_DIR / "train_segment_label_report.csv",
            created,
        )
    else:
        _warn("Train segment label report skipped; requires train scores, train features, and labels.")

    _model_card(final_model, thresholds, created)

    _write_json({"warnings": WARNINGS}, OUTPUT_DIR / "week4_warnings.json", created)

    print("\nCreated/updated files:")
    for path in created:
        print(f"- {path.relative_to(PROJECT_ROOT)}")
    if WARNINGS:
        print("\nWarnings:")
        for warning in WARNINGS:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
