# VDT_VDS_2026 - Device Trust Score

Device Trust Score (DTS) is a fraud-risk and device-trust scoring project for telecom/device data. The final scoring path uses supervised Track B probabilities (`P_fraud`) calibrated into a 0-1000 DTS scale, where higher DTS means more trusted and higher `P_fraud` means higher risk.

## Project Overview

The project supports three business use cases:

- KYC: fast eKYC for trusted customers, step-up or manual review for risky customers.
- Fraud Detection: rank customers by fraud risk with review capacity capped around 5%.
- Credit Input: use DTS as an auxiliary credit-model feature, not as a standalone auto-reject rule.

Final production-friendly model choice is `xgb_tuned`. Current summary metrics indicate XGB is nearly tied with stacking while being simpler to deploy and explain.

## Folder Structure

```text
data/                  Raw input CSV files
docs/                  Reports
notebooks/             EDA, unsupervised, clustering, supervised modeling
outputs/               Generated features, submissions, ...
scripts/               Reproducible command-line scripts
src/                   Feature, model, scorecard, evaluation, business modules
```

## Setup Environment

Requirements:

- Python 3.11
- `uv`

```bash
uv python pin 3.11
uv venv --python 3.11
source .venv/bin/activate
uv sync
python -m ipykernel install --user --name vdt-vds-2026 --display-name "Python (VDT_VDS_2026)"
```

On macOS, LightGBM/XGBoost may need OpenMP:

```bash
brew install libomp
```

## Data Placement

Place these files under `data/`:

- `dts_train.csv`
- `dts_holdout.csv`
- `sim_events.csv`
- `device_sessions.csv`
- `kyc_records.csv`
- `device_catalog.csv`

Do not add answer-key columns to holdout feature generation. Train labels must only come from `dts_train.csv`.

## How to Run

1. Create feature snapshots and Track A artifacts with notebooks:

```text
notebooks/EDA.ipynb
notebooks/unsupervised.ipynb
notebooks/clustering.ipynb
notebooks/dts_unsup.ipynb
```

2. Build Track A component submissions and optional `sim_swap_ato` diagnostics:

```bash
python scripts/tracka_component_analysis.py
```

This creates `outputs/unsup_submit/` with separate ISO, LOF, blend, and graph-adjusted submission files. It also writes `tracka_component_sim_swap_ato_metrics.csv`.

If a holdout answer key is available, place it at one of these paths with columns `CustomerID`, `FraudFlag`, `FraudType`:

```text
data/dts_holdout_labeled.csv
data/dts_holdout_answer.csv
data/dts_holdout_answer_key.csv
data/holdout_answer_key.csv
outputs/dts_holdout_labeled.csv
outputs/holdout_answer_key.csv
outputs/trackA_holdout_answer_key.csv
```

Without that label file, the script still creates the submission CSVs and marks component AUC/recall as `missing_holdout_labels` instead of inventing numbers.

3. Run supervised modeling:

```text
notebooks/supervised.ipynb
```

This should create files under `outputs/supervised/`, including submissions in `outputs/supervised/submit/`, calibrated holdout scores, SHAP importance, and reason codes.

4. Generate train OOF scores for score PSI:

```bash
python scripts/generate_train_oof_scores.py
```

This creates `outputs/supervised/train_oof_scores.csv` from the notebook-equivalent `xgb_tuned` OOF flow.

5. Finalize week 4 artifacts:

```bash
python scripts/week4_finalize.py
```

If the active environment has parquet support, the script can read engineered feature snapshots.

## Expected Outputs

Week 4 finalization creates or updates:

- `outputs/unsup_submit/tracka_iso_submission.csv`
- `outputs/unsup_submit/tracka_lof_submission.csv`
- `outputs/unsup_submit/tracka_blend_submission.csv`
- `outputs/unsup_submit/tracka_graph_adjusted_submission.csv`
- `outputs/unsup_submit/tracka_component_sim_swap_ato_metrics.csv`
- `outputs/supervised/holdout_submission_final.csv`
- `outputs/supervised/decision_matrix.csv`
- `outputs/supervised/holdout_decision_actions.csv`
- `outputs/supervised/decision_thresholds.json`
- `outputs/supervised/train_oof_scores.csv`
- `outputs/supervised/score_psi_report.csv`
- `outputs/supervised/feature_psi_report.csv`
- `outputs/supervised/fairness_segment_report.csv`
- `outputs/supervised/final_model_card.md`
- `outputs/supervised/week4_warnings.json`
- `docs/VDT_final_report.typ`

The final submission file is:

```text
outputs/supervised/holdout_submission_final.csv
```

## Reproducibility Notes

- Restart kernels and run notebooks from top to bottom.
- Keep random seeds fixed where notebooks define them.
- Do not leak holdout answer-key columns into feature engineering.
