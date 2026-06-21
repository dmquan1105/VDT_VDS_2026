# VDT_VDS_2026

Device Trust Score (DTS) – Fraud Detection & Unsupervised Risk Scoring

## Môi trường

### Requirements

- Python 3.11
- uv

### Tạo môi trường

```bash
uv python pin 3.11
uv venv --python 3.11
source .venv/bin/activate
```

(Môi trường này dùng trong cả notebooks)

```bash
python -m ipykernel install --user --name vdt-vds-2026 --display-name "Python (VDT_VDS_2026)"
```

### Install dependencies

```bash
uv sync
```

Hoặc:

```bash
uv add pandas numpy scikit-learn matplotlib seaborn jupyter pyarrow plotly umap-learn
```

## Data

Input tables:

- dts_train.csv
- dts_holdout.csv
- sim_events.csv
- device_sessions.csv
- kyc_records.csv
- device_catalog.csv

---

## Reproducibility

```bash
uv sync
```

sẽ tạo môi trường đúng, sử dụng:

```text
pyproject.toml
uv.lock
```
