"""
src/advanced.py

Utilities for multi-series and covariate experiments
with Chronos-2.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from darts import TimeSeries


# ============================================================
# CONSTANTS
# ============================================================

TIMESTAMP_COLUMN = "timestamp"


# ============================================================
# SERIES DISCOVERY
# ============================================================

def get_series_columns(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COLUMN,
) -> List[str]:
    """
    Return all electricity-series columns.

    All columns except the timestamp are treated as
    target series.
    """

    return [
        column
        for column in df.columns
        if column != timestamp_col
    ]


# ============================================================
# SELECT SERIES
# ============================================================

def select_series_ids(
    df: pd.DataFrame,
    series_ids: Optional[Sequence[str]] = None,
    timestamp_col: str = TIMESTAMP_COLUMN,
) -> List[str]:
    """
    Select target series.

    If series_ids is None, return all available series.
    """

    available = get_series_columns(
        df,
        timestamp_col=timestamp_col,
    )

    if series_ids is None:
        return available

    missing = [
        series_id
        for series_id in series_ids
        if series_id not in available
    ]

    if missing:
        raise ValueError(
            f"Unknown series IDs: {missing}"
        )

    return list(series_ids)


# ============================================================
# DATAFRAME → MULTIPLE DARTS SERIES
# ============================================================

def dataframe_to_multiple_series(
    df: pd.DataFrame,
    series_ids: Optional[Sequence[str]] = None,
    timestamp_col: str = TIMESTAMP_COLUMN,
    fill_missing: bool = False,
) -> Dict[str, TimeSeries]:
    """
    Convert multiple electricity columns into a dictionary
    of Darts TimeSeries.

    Returns
    -------
    dict
        {
            "MT_001": TimeSeries,
            "MT_002": TimeSeries,
            ...
        }
    """

    selected = select_series_ids(
        df,
        series_ids=series_ids,
        timestamp_col=timestamp_col,
    )

    data = df.copy()

    data[timestamp_col] = pd.to_datetime(
        data[timestamp_col]
    )

    data = data.sort_values(
        timestamp_col
    ).reset_index(drop=True)

    result = {}

    for series_id in selected:

        series_df = data[
            [
                timestamp_col,
                series_id,
            ]
        ].copy()

        if fill_missing:

            series_df[series_id] = (
                series_df[series_id]
                .interpolate()
                .ffill()
                .bfill()
            )

        result[series_id] = (
            TimeSeries
            .from_dataframe(
                series_df,
                time_col=timestamp_col,
                value_cols=series_id,
                fill_missing_dates=False,
            )
        )

    return result


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split_multiple_series(
    series_dict: Dict[str, TimeSeries],
    train_ratio: float = 0.8,
) -> Tuple[
    Dict[str, TimeSeries],
    Dict[str, TimeSeries],
]:
    """
    Chronologically split every series using the same ratio.
    """

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    train = {}
    test = {}

    for series_id, series in series_dict.items():

        train_series, test_series = (
            series.split_before(train_ratio)
        )

        train[series_id] = train_series
        test[series_id] = test_series

    return train, test


# ============================================================
# SERIES LENGTH CHECK
# ============================================================

def check_series_alignment(
    series_dict: Dict[str, TimeSeries],
) -> pd.DataFrame:
    """
    Check start/end time and length for multiple series.
    """

    rows = []

    for series_id, series in series_dict.items():

        rows.append(
            {
                "series_id": series_id,
                "length": len(series),
                "start": series.start_time(),
                "end": series.end_time(),
                "frequency": series.freq,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SELECT TOP SERIES BY VARIANCE
# ============================================================

def select_top_series_by_variance(
    series_dict: Dict[str, TimeSeries],
    n: int = 5,
) -> List[str]:
    """
    Select the n series with highest variance.

    Useful for a manageable demonstration experiment.
    """

    variances = {}

    for series_id, series in series_dict.items():

        values = series.values(copy=False).flatten()

        variances[series_id] = np.var(
            values
        )

    return [
        series_id
        for series_id, _ in sorted(
            variances.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]
    ]


# ============================================================
# TIME COVARIATES
# ============================================================

def create_time_covariates(
    series: TimeSeries,
) -> TimeSeries:
    """
    Create deterministic calendar/time covariates.

    Features:
        - hour
        - day_of_week
        - day_of_month
        - month
        - is_weekend

    These are deterministic and therefore available
    for future timestamps.
    """

    time_index = series.time_index

    covariates = pd.DataFrame(
        {
            "timestamp": time_index,
            "hour": time_index.hour,
            "day_of_week": time_index.dayofweek,
            "day_of_month": time_index.day,
            "month": time_index.month,
            "is_weekend": (
                time_index.dayofweek >= 5
            ).astype(int),
        }
    )

    return TimeSeries.from_dataframe(
        covariates,
        time_col="timestamp",
        value_cols=[
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
        ],
    )


# ============================================================
# CYCLIC TIME COVARIATES
# ============================================================

def create_cyclic_time_covariates(
    series: TimeSeries,
) -> TimeSeries:
    """
    Create cyclic encodings for calendar variables.

    Cyclic encoding avoids treating:
        23:00 and 00:00
    as very far apart.
    """

    time_index = series.time_index

    hour = time_index.hour

    day_of_week = time_index.dayofweek

    month = time_index.month

    covariates = pd.DataFrame(
        {
            "timestamp": time_index,

            "hour_sin": np.sin(
                2 * np.pi * hour / 24
            ),

            "hour_cos": np.cos(
                2 * np.pi * hour / 24
            ),

            "dow_sin": np.sin(
                2 * np.pi * day_of_week / 7
            ),

            "dow_cos": np.cos(
                2 * np.pi * day_of_week / 7
            ),

            "month_sin": np.sin(
                2 * np.pi * (month - 1) / 12
            ),

            "month_cos": np.cos(
                2 * np.pi * (month - 1) / 12
            ),
        }
    )

    return TimeSeries.from_dataframe(
        covariates,
        time_col="timestamp",
        value_cols=[
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ],
    )


# ============================================================
# ADD FUTURE TIMESTAMPS
# ============================================================

def extend_time_covariates(
    covariates: TimeSeries,
    future_end: pd.Timestamp,
    freq: Optional[str] = None,
) -> TimeSeries:
    """
    Extend deterministic time covariates into the future.

    This is useful because future covariates must cover
    the forecast horizon.
    """

    if freq is None:
        freq = covariates.freq_str

    future_index = pd.date_range(
        start=covariates.end_time(),
        end=future_end,
        freq=freq,
    )

    full_index = covariates.time_index.union(
        future_index
    )

    dummy = TimeSeries.from_times_and_values(
        times=full_index,
        values=np.zeros(
            (len(full_index), 1)
        ),
    )

    return create_cyclic_time_covariates(
        dummy
    )


# ============================================================
# FUTURE COVARIATE BUILDER
# ============================================================

def create_future_cyclic_covariates(
    series: TimeSeries,
    forecast_horizon: int,
) -> TimeSeries:
    """
    Create cyclic covariates covering both historical
    context and the requested future horizon.
    """

    freq = series.freq_str

    full_index = pd.date_range(
        start=series.start_time(),
        periods=len(series) + forecast_horizon,
        freq=freq,
    )

    dummy = TimeSeries.from_times_and_values(
        times=full_index,
        values=np.zeros(
            (len(full_index), 1)
        ),
    )

    return create_cyclic_time_covariates(
        dummy
    )


# ============================================================
# CONVERT DICTIONARY TO LIST
# ============================================================

def series_dict_to_list(
    series_dict: Dict[str, TimeSeries],
) -> List[TimeSeries]:
    """
    Convert dictionary of TimeSeries to list.

    Useful for Darts APIs accepting multiple series.
    """

    return list(
        series_dict.values()
    )


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

def summarize_multiple_series(
    series_dict: Dict[str, TimeSeries],
) -> pd.DataFrame:
    """
    Produce basic statistics for each selected series.
    """

    rows = []

    for series_id, series in series_dict.items():

        values = series.values(
            copy=False
        ).flatten()

        rows.append(
            {
                "Series_ID": series_id,
                "Observations": len(values),
                "Minimum": np.min(values),
                "Maximum": np.max(values),
                "Mean": np.mean(values),
                "Std": np.std(values),
                "Missing": np.isnan(values).sum(),
            }
        )

    return pd.DataFrame(rows)