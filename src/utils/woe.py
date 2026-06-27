from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


MISSING_BUCKET = "__MISSING__" # NaN, None
RARE_BUCKET = "__RARE__" # Category < min_samples -> RARE


@dataclass
class WOEColumnMapping:
    feature_name: str
    mapping: dict[str, float]
    rare_values: set[str] = field(default_factory=set)
    default_woe: float = 0.0


class WOEEncoder:
    """
    Weight-of-evidence encoder for target-aware categorical preprocessing.

    Fit this encoder only on the training data or inside each CV fold. The
    transform step can then be applied to validation/holdout data with unseen
    categories falling back to the learned rare bucket or 0.0.
    """

    def __init__(
        self,
        event_label: int = 1,
        smoothing: float = 0.5,
        min_samples: int = 20,
        suffix: str = "_woe",
    ) -> None:
        self.event_label = event_label
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.suffix = suffix
        self.columns_: list[str] = []
        self.mappings_: dict[str, WOEColumnMapping] = {}
        self.mapping_frame_: pd.DataFrame | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        columns: list[str] | None = None,
    ) -> "WOEEncoder":
        if columns is None:
            columns = X.select_dtypes(
                include=["object", "string", "category"]
            ).columns.to_list()

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        y_binary = (pd.Series(y).reset_index(drop=True) == self.event_label).astype(int)
        total_events = int(y_binary.sum())
        total_non_events = int(len(y_binary) - total_events)
        if total_events == 0 or total_non_events == 0:
            raise ValueError("WOE requires both event and non-event observations")

        self.columns_ = list(columns)
        self.mappings_ = {}
        mapping_rows = []

        for col in self.columns_:
            prepared = self._prepare_series(X[col])
            value_counts = prepared.value_counts(dropna=False)
            rare_values = set(
                value_counts[value_counts < self.min_samples].index.astype(str)
            )

            bucketed = prepared.mask(prepared.isin(rare_values), RARE_BUCKET)
            stats = (
                pd.DataFrame({"bucket": bucketed, "target": y_binary})
                .groupby("bucket", dropna=False)["target"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "event_count", "count": "total_count"})
                .reset_index()
            )
            stats["non_event_count"] = stats["total_count"] - stats["event_count"]
            bucket_count = len(stats)

            stats["event_dist"] = (
                stats["event_count"] + self.smoothing
            ) / (total_events + self.smoothing * bucket_count)
            stats["non_event_dist"] = (
                stats["non_event_count"] + self.smoothing
            ) / (total_non_events + self.smoothing * bucket_count)
            stats["woe"] = np.log(stats["non_event_dist"] / stats["event_dist"])
            stats["event_rate"] = stats["event_count"] / stats["total_count"]

            mapping = dict(zip(stats["bucket"].astype(str), stats["woe"].astype(float)))
            default_woe = mapping.get(RARE_BUCKET, 0.0)
            self.mappings_[col] = WOEColumnMapping(
                feature_name=col,
                mapping=mapping,
                rare_values=rare_values,
                default_woe=default_woe,
            )

            for row in stats.to_dict("records"):
                mapping_rows.append(
                    {
                        "feature_name": col,
                        "category": row["bucket"],
                        "total_count": int(row["total_count"]),
                        "event_count": int(row["event_count"]),
                        "non_event_count": int(row["non_event_count"]),
                        "event_rate": float(row["event_rate"]),
                        "woe": float(row["woe"]),
                        "is_rare_bucket": row["bucket"] == RARE_BUCKET,
                        "is_missing_bucket": row["bucket"] == MISSING_BUCKET,
                    }
                )

        self.mapping_frame_ = pd.DataFrame(mapping_rows)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.mappings_:
            raise ValueError("WOEEncoder must be fitted before transform")

        encoded = pd.DataFrame(index=X.index)
        for col in self.columns_:
            column_mapping = self.mappings_[col]
            prepared = self._prepare_series(X[col])
            bucketed = prepared.mask(
                prepared.isin(column_mapping.rare_values),
                RARE_BUCKET,
            )
            encoded[f"{col}{self.suffix}"] = (
                bucketed.map(column_mapping.mapping)
                .fillna(column_mapping.default_woe)
                .astype(float)
            )

        return encoded

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.fit(X, y, columns=columns).transform(X)

    @staticmethod
    def _prepare_series(series: pd.Series) -> pd.Series:
        return (
            series.astype("string")
            .fillna(MISSING_BUCKET)
            .replace("", MISSING_BUCKET)
            .astype(str)
        )
