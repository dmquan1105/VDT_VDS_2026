"""Decision policy mapping for Device Trust Score outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_decision_matrix(thresholds: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return the business decision matrix using distribution-based thresholds."""
    thresholds = thresholds or {}

    def fmt(name: str) -> str: # format
        value = thresholds.get(name)
        if value is None:
            return "computed_from_holdout_distribution"
        return f"{float(value):.8f}".rstrip("0").rstrip(".")

    rows = [
        {
            "use_case": "KYC",
            "score_driver": "DTS",
            "policy_segment": "Top 30% DTS",
            "rule": f"DTS >= kyc_fast_cutoff ({fmt('kyc_fast_cutoff')})",
            "action": "fast_ekyc",
            "rationale": "DTS cao là đáng tin; ngưỡng đề xuất dựa trên phân phối điểm holdout.",
        },
        {
            "use_case": "KYC",
            "score_driver": "DTS",
            "policy_segment": "Bottom 5% DTS",
            "rule": f"DTS <= kyc_reject_cutoff ({fmt('kyc_reject_cutoff')})",
            "action": "reject_or_enhanced_review",
            "rationale": "Nhóm DTS thấp nhất cần enhanced review; business có thể chỉnh theo khẩu vị rủi ro.",
        },
        {
            "use_case": "KYC",
            "score_driver": "DTS",
            "policy_segment": "Middle DTS",
            "rule": "otherwise",
            "action": "additional_verification",
            "rationale": "Không đủ tin cậy để fast eKYC, nhưng cũng chưa phải nhóm reject/enhanced-review.",
        },
        {
            "use_case": "Fraud Detection",
            "score_driver": "P_fraud",
            "policy_segment": "Top 0.5% P_fraud",
            "rule": f"P_fraud >= fraud_decline_cutoff_p ({fmt('fraud_decline_cutoff_p')})",
            "action": "decline_or_block",
            "rationale": "Nhóm cực rủi ro; cần strong fraud signals và phê duyệt business trước khi block.",
        },
        {
            "use_case": "Fraud Detection",
            "score_driver": "P_fraud",
            "policy_segment": "Top 5% P_fraud excluding decline",
            "rule": f"P_fraud >= fraud_review_cutoff_p ({fmt('fraud_review_cutoff_p')})",
            "action": "manual_review",
            "rationale": "Dùng P_fraud để giữ đúng operating point review khoảng 5%.",
        },
        {
            "use_case": "Fraud Detection",
            "score_driver": "P_fraud",
            "policy_segment": "Top 20% P_fraud excluding review",
            "rule": f"P_fraud >= fraud_stepup_cutoff_p ({fmt('fraud_stepup_cutoff_p')})",
            "action": "step_up_authentication",
            "rationale": "Rủi ro tăng nhưng chưa vào hàng đợi manual review.",
        },
        {
            "use_case": "Fraud Detection",
            "score_driver": "P_fraud",
            "policy_segment": "Lower 80% P_fraud",
            "rule": "otherwise",
            "action": "approve",
            "rationale": "Không có tín hiệu đủ mạnh để step-up hoặc review theo policy hiện tại.",
        },
        {
            "use_case": "Credit Input",
            "score_driver": "DTS",
            "policy_segment": "Top 30% DTS",
            "rule": f"DTS >= credit_high_cutoff ({fmt('credit_high_cutoff')})",
            "action": "normal_credit_flow_with_low_device_risk",
            "rationale": "DTS chỉ là feature bổ sung/risk overlay, không thay thế credit score.",
        },
        {
            "use_case": "Credit Input",
            "score_driver": "DTS",
            "policy_segment": "Bottom 10% DTS",
            "rule": f"DTS <= credit_low_cutoff ({fmt('credit_low_cutoff')})",
            "action": "fraud_review_or_step_up_before_credit_decision",
            "rationale": "Không auto reject tín dụng một mình; cần xử lý fraud/device-risk trước quyết định credit.",
        },
        {
            "use_case": "Credit Input",
            "score_driver": "DTS",
            "policy_segment": "Middle DTS",
            "rule": "otherwise",
            "action": "use_dts_as_auxiliary_credit_feature",
            "rationale": "DTS được dùng như biến phụ trợ trong credit model hoặc risk overlay.",
        },
    ]
    return pd.DataFrame(rows)


def assign_decision_actions(
    scored_df: pd.DataFrame,
    review_rate: float = 0.05,
    extreme_rate: float = 0.005,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign KYC, fraud, and credit actions to scored customers."""
    required = {"CustomerID", "P_fraud", "DTS"}
    missing = required.difference(scored_df.columns)
    if missing:
        raise ValueError(f"Missing required scored columns: {sorted(missing)}")
    if not 0 < review_rate < 1:
        raise ValueError("review_rate must be between 0 and 1")
    if not 0 < extreme_rate < 1:
        raise ValueError("extreme_rate must be between 0 and 1")

    result = scored_df.copy()
    result["P_fraud"] = pd.to_numeric(result["P_fraud"], errors="coerce")
    result["DTS"] = pd.to_numeric(result["DTS"], errors="coerce")
    if result[["P_fraud", "DTS"]].isna().any().any():
        raise ValueError("P_fraud and DTS must be numeric and non-missing")

    kyc_fast_cutoff = float(result["DTS"].quantile(0.70))
    kyc_reject_cutoff = float(result["DTS"].quantile(0.05))
    fraud_review_cutoff_p = float(result["P_fraud"].quantile(1 - review_rate))
    fraud_decline_cutoff_p = float(result["P_fraud"].quantile(1 - extreme_rate))
    fraud_stepup_cutoff_p = float(result["P_fraud"].quantile(0.80))
    fraud_review_cutoff_dts_q05 = float(result["DTS"].quantile(review_rate))
    credit_low_cutoff = float(result["DTS"].quantile(0.10))
    credit_high_cutoff = float(result["DTS"].quantile(0.70))

    result["kyc_action"] = "additional_verification"
    result.loc[result["DTS"] >= kyc_fast_cutoff, "kyc_action"] = "fast_ekyc"
    result.loc[result["DTS"] <= kyc_reject_cutoff, "kyc_action"] = "reject_or_enhanced_review"

    result["fraud_action"] = "approve"
    result.loc[result["P_fraud"] >= fraud_stepup_cutoff_p, "fraud_action"] = "step_up_authentication"
    result.loc[result["P_fraud"] >= fraud_review_cutoff_p, "fraud_action"] = "manual_review"
    result.loc[result["P_fraud"] >= fraud_decline_cutoff_p, "fraud_action"] = "decline_or_block"

    result["credit_treatment"] = "use_dts_as_auxiliary_credit_feature"
    result.loc[
        result["DTS"] >= credit_high_cutoff,
        "credit_treatment",
    ] = "normal_credit_flow_with_low_device_risk"
    result.loc[
        result["DTS"] <= credit_low_cutoff,
        "credit_treatment",
    ] = "fraud_review_or_step_up_before_credit_decision"

    thresholds = {
        "kyc_fast_cutoff": kyc_fast_cutoff,
        "kyc_reject_cutoff": kyc_reject_cutoff,
        "fraud_review_cutoff_p": fraud_review_cutoff_p,
        "fraud_decline_cutoff_p": fraud_decline_cutoff_p,
        "fraud_stepup_cutoff_p": fraud_stepup_cutoff_p,
        "fraud_review_cutoff_dts_q05": fraud_review_cutoff_dts_q05,
        "credit_low_cutoff": credit_low_cutoff,
        "credit_high_cutoff": credit_high_cutoff,
        "expected_review_rate": review_rate,
        "expected_extreme_rate": extreme_rate,
        "manual_review_rate": float((result["fraud_action"] == "manual_review").mean()),
        "decline_or_block_rate": float((result["fraud_action"] == "decline_or_block").mean()),
        "manual_review_or_decline_rate": float(
            result["fraud_action"].isin(["manual_review", "decline_or_block"]).mean()
        ),
        "step_up_authentication_rate": float((result["fraud_action"] == "step_up_authentication").mean()),
        "approve_rate": float((result["fraud_action"] == "approve").mean()),
        "total_fraud_intervention_rate": float((result["fraud_action"] != "approve").mean()),
        "kyc_fast_rate": float((result["kyc_action"] == "fast_ekyc").mean()),
        "kyc_reject_or_enhanced_review_rate": float(
            (result["kyc_action"] == "reject_or_enhanced_review").mean()
        ),
        "credit_low_device_risk_rate": float(
            (result["credit_treatment"] == "normal_credit_flow_with_low_device_risk").mean()
        ),
        "credit_fraud_review_or_step_up_rate": float(
            (result["credit_treatment"] == "fraud_review_or_step_up_before_credit_decision").mean()
        ),
    }
    return result, thresholds
