from pathlib import Path
import pandas as pd


def load_raw_data(base_dir: str | Path = "../data") -> dict:
    base_dir = Path(base_dir)

    return {
        "device_catalog": pd.read_csv(base_dir / "device_catalog.csv"),
        "device_sessions": pd.read_csv(base_dir / "device_sessions.csv"),
        "dts_holdout": pd.read_csv(base_dir / "dts_holdout.csv"),
        "dts_train": pd.read_csv(base_dir / "dts_train.csv"),
        "kyc_records": pd.read_csv(base_dir / "kyc_records.csv"),
        "sim_events": pd.read_csv(base_dir / "sim_events.csv"),
    }


def save_figure(fig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)