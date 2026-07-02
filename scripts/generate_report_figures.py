"""Generate PNG report figures from saved output tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "figures"
SUPERVISED_DIR = PROJECT_ROOT / "outputs" / "supervised"


COLORS = {
    "blue": "#2f6f9f",
    "teal": "#3a8f7a",
    "red": "#b85c5c",
    "gold": "#c08a2b",
    "gray": "#6b7280",
    "purple": "#7c5fb3",
}


def _save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_track_summary_metrics() -> None:
    df = pd.DataFrame(
        {
            "Pipeline": ["Baseline ISO", "Blend LOF+ISO+graph", "Track B XGB"],
            "AUC": [0.777, 0.789, 0.954],
            "PR-AUC": [0.391, 0.302, 0.797],
            "Precision@5%": [0.280, 0.227, 0.508],
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    df.set_index("Pipeline").plot(
        kind="bar",
        ax=ax,
        color=[COLORS["blue"], COLORS["teal"], COLORS["gold"]],
        width=0.72,
    )
    ax.set_title("Track Summary Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=0)
    _style_axes(ax)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    _save(fig, "track_summary_metrics.png")


def plot_supervised_model_metrics() -> None:
    df = pd.DataFrame(
        {
            "Model": ["xgb_tuned", "stacking_tuned", "gbm_tuned", "lgbm_tuned", "logistic_tuned"],
            "PR-AUC": [0.797, 0.797, 0.792, 0.790, 0.673],
            "Precision@5%": [0.508, 0.510, 0.410, 0.463, 0.445],
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    df.set_index("Model").plot(
        kind="bar",
        ax=ax,
        color=[COLORS["teal"], COLORS["gold"]],
        width=0.72,
    )
    ax.set_title("Supervised Model Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.9)
    ax.tick_params(axis="x", rotation=20)
    _style_axes(ax)
    ax.legend(frameon=False)
    _save(fig, "supervised_model_metrics.png")


def plot_xgb_reliability_curve() -> None:
    path = SUPERVISED_DIR / "supervised_calibration_reliability_curve.csv"
    df = pd.read_csv(path)
    plot_df = (
        df[df["model_name"].isin(["xgb_tuned", "xgb"])]
        .dropna(subset=["mean_predicted_prob", "mean_calibrated_prob", "actual_fraud_rate"])
        .sort_values("mean_predicted_prob")
    )
    if plot_df.empty:
        raise ValueError(f"No xgb_tuned reliability rows found in {path}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], linewidth=1.4, label="Perfect calibration")
    ax.plot(
        plot_df["mean_predicted_prob"],
        plot_df["actual_fraud_rate"],
        marker="o",
        linewidth=2.0,
        color=COLORS["blue"],
        label="Raw mean predicted probability",
    )
    ax.plot(
        plot_df["mean_calibrated_prob"],
        plot_df["actual_fraud_rate"],
        marker="s",
        linewidth=2.0,
        color=COLORS["teal"],
        label="Calibrated mean predicted probability",
    )
    ax.set_title("XGB Reliability Curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual fraud rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "xgb_reliability_curve.png")


def plot_decision_action_rates() -> None:
    df = pd.read_csv(SUPERVISED_DIR / "holdout_decision_actions.csv")
    order = ["approve", "step_up_authentication", "manual_review", "decline_or_block"]
    rates = df["fraud_action"].value_counts(normalize=True).reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(rates.index, rates.values, color=[COLORS["teal"], COLORS["gold"], COLORS["blue"], COLORS["red"]])
    ax.set_title("Fraud Action Rates")
    ax.set_xlabel("")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(0.85, rates.max() * 1.2))
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.tick_params(axis="x", rotation=15)
    _style_axes(ax)
    _save(fig, "decision_action_rates.png")


def plot_feature_psi_top() -> None:
    df = pd.read_csv(SUPERVISED_DIR / "feature_psi_report.csv")
    plot_df = df.sort_values("psi", ascending=False).head(10).sort_values("psi")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(plot_df["variable"], plot_df["psi"], color=COLORS["teal"])
    ax.set_title("Top Feature PSI")
    ax.set_xlabel("PSI")
    ax.set_ylabel("")
    _style_axes(ax)
    _save(fig, "feature_psi_top.png")


def plot_fairness_review_rates() -> None:
    df = pd.read_csv(SUPERVISED_DIR / "fairness_segment_report.csv")
    rate_col = "review_or_block_rate" if "review_or_block_rate" in df.columns else "review_rate"
    plot_df = df.sort_values(rate_col)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(plot_df["segment"], plot_df[rate_col], color=COLORS["purple"])
    ax.set_title("Review/Block Rate by Segment")
    ax.set_xlabel("Review/block rate")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    _style_axes(ax)
    _save(fig, "fairness_review_rates.png")


def plot_shap_top_features() -> None:
    df = pd.read_csv(SUPERVISED_DIR / "supervised_global_shap_importance.csv")
    value_col = "mean_abs_shap" if "mean_abs_shap" in df.columns else "mean_abs_shap_value"
    plot_df = df.sort_values(value_col, ascending=False).head(10).sort_values(value_col)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(plot_df["feature"], plot_df[value_col], color=COLORS["blue"])
    ax.set_title("Top SHAP Features")
    ax.set_xlabel("Mean absolute SHAP")
    ax.set_ylabel("")
    _style_axes(ax)
    _save(fig, "shap_top_features.png")


def main() -> None:
    plot_track_summary_metrics()
    plot_supervised_model_metrics()
    plot_xgb_reliability_curve()
    plot_decision_action_rates()
    plot_feature_psi_top()
    plot_fairness_review_rates()
    plot_shap_top_features()


if __name__ == "__main__":
    main()
