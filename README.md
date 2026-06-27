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

### Lưu ý macOS cho LightGBM/XGBoost

LightGBM và XGBoost dùng native binary cần OpenMP runtime. Trên macOS, nếu notebook báo thiếu `libomp.dylib`, cài thêm:

```bash
brew install libomp
```

Sau đó restart kernel/notebook và chạy lại. Nếu vẫn lỗi import, reinstall hai package trong môi trường hiện tại:

```bash
uv pip install --force-reinstall lightgbm xgboost
```

Nếu chưa cài `libomp`, các notebook vẫn có thể chạy Logistic Regression và sklearn Gradient Boosting; phần LightGBM/XGBoost sẽ được skip.

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

## Thứ tự notebook week 2:

`unsupervised.ipynb` -> `clustering.ipynb` -> `dts_unsup.ipynb`
