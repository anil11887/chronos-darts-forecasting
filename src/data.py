"""
src/data.py

Reusable data loading and inspection utilities
for the Chronos-2 / Darts forecasting project.

Dataset:
    data/raw/LD2011_2014.txt

Expected raw structure:
    timestamp;MT_001;MT_002;MT_003;...

Converted structure:
    timestamp | series_id | value
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATA_PATH = "data/raw/LD2011_2014.txt"

TIMESTAMP_COL = "timestamp"
SERIES_COL = "series_id"
VALUE_COL = "value"


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

def load_raw_dataset(
    file_path: str = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the original LD2011_2014.txt dataset.

    The dataset is expected to be semicolon-separated.

    Returns
    -------
    pd.DataFrame
        Raw wide-format dataframe.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path.resolve()}"
        )

    print(f"Loading dataset: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
        low_memory=False,
        index_col=0,
    )

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: TIMESTAMP_COL})  # rename by position, not by name

    print(f"Raw shape: {df.shape}")

    return df

# ============================================================
# 2. CLEAN TIMESTAMP
# ============================================================

def clean_timestamp(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """
    Convert timestamp column to pandas datetime.
    """

    df = df.copy()

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=[timestamp_col]
    )

    return df


# ============================================================
# 3. CONVERT WIDE → LONG
# ============================================================

def convert_to_long_format(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    series_col: str = SERIES_COL,
    value_col: str = VALUE_COL,
) -> pd.DataFrame:
    """
    Convert the electricity dataset from wide format
    to long format.

    Wide:

        timestamp | MT_001 | MT_002 | MT_003

    Long:

        timestamp | series_id | value
    """

    df = clean_timestamp(
        df,
        timestamp_col
    )

    # Every column except timestamp represents a series
    series_columns = [
        col for col in df.columns
        if col != timestamp_col
    ]

    long_df = df.melt(
        id_vars=[timestamp_col],
        value_vars=series_columns,
        var_name=series_col,
        value_name=value_col
    )

    # Ensure numeric values
    long_df[value_col] = pd.to_numeric(
        long_df[value_col],
        errors="coerce"
    )

    # Sort
    long_df = long_df.sort_values(
        by=[series_col, timestamp_col]
    ).reset_index(drop=True)

    return long_df


# ============================================================
# 4. LOAD PROCESSED DATASET
# ============================================================

def load_dataset(
    file_path: str = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Complete dataset loading pipeline.

    Returns long-format dataframe:

        timestamp | series_id | value
    """

    raw_df = load_raw_dataset(
        file_path
    )

    df = convert_to_long_format(
        raw_df
    )

    return df


# ============================================================
# 5. BASIC STATISTICS
# ============================================================

def get_basic_statistics(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate basic dataset statistics.

    Answers:
        - How many observations?
        - Number of series?
        - Minimum value?
        - Maximum value?
        - Date range?
    """

    observations = len(df)

    number_of_series = df[
        SERIES_COL
    ].nunique()

    minimum = df[
        VALUE_COL
    ].min()

    maximum = df[
        VALUE_COL
    ].max()

    start_date = df[
        TIMESTAMP_COL
    ].min()

    end_date = df[
        TIMESTAMP_COL
    ].max()

    return {
        "observations": observations,
        "number_of_series": number_of_series,
        "minimum": minimum,
        "maximum": maximum,
        "start_date": start_date,
        "end_date": end_date,
    }


# ============================================================
# 6. FREQUENCY DETECTION
# ============================================================

def detect_frequency(
    df: pd.DataFrame,
) -> Optional[str]:
    """
    Detect the frequency of the time series.

    Uses timestamp differences and pandas inference.

    Returns
    -------
    str or None
        Frequency such as:

        H
        15min
        D
        W
        etc.
    """

    frequencies = []

    # Check each series separately
    for series_id, group in df.groupby(
        SERIES_COL
    ):

        timestamps = (
            group[TIMESTAMP_COL]
            .drop_duplicates()
            .sort_values()
        )

        if len(timestamps) < 3:
            continue

        try:
            freq = pd.infer_freq(
                timestamps
            )

            if freq is not None:
                frequencies.append(freq)

        except ValueError:
            continue

    if not frequencies:
        return None

    # Most common frequency
    frequency = (
        pd.Series(frequencies)
        .value_counts()
        .idxmax()
    )

    return frequency


# ============================================================
# 7. TIME INTERVAL ANALYSIS
# ============================================================

def get_time_intervals(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate time differences between
    consecutive observations.

    Useful when infer_freq() cannot determine
    the frequency.
    """

    intervals = []

    for _, group in df.groupby(
        SERIES_COL
    ):

        timestamps = (
            group[TIMESTAMP_COL]
            .drop_duplicates()
            .sort_values()
        )

        differences = (
            timestamps
            .diff()
            .dropna()
        )

        intervals.extend(
            differences.tolist()
        )

    return pd.Series(
        intervals
    )


# ============================================================
# 8. MISSING VALUE CHECK
# ============================================================

def check_missing_values(
    df: pd.DataFrame,
) -> dict:
    """
    Check missing values and duplicate
    timestamp-series combinations.
    """

    missing_by_column = (
        df[
            [
                TIMESTAMP_COL,
                SERIES_COL,
                VALUE_COL
            ]
        ]
        .isnull()
        .sum()
        .to_dict()
    )

    missing_values = int(
        df[VALUE_COL]
        .isnull()
        .sum()
    )

    duplicate_timestamps = int(
        df.duplicated(
            subset=[
                TIMESTAMP_COL,
                SERIES_COL
            ]
        ).sum()
    )

    return {
        "missing_by_column": missing_by_column,
        "missing_value_count": missing_values,
        "duplicate_timestamp_series": duplicate_timestamps,
    }


# ============================================================
# 9. PER-SERIES STATISTICS
# ============================================================

def get_series_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate statistics for every series.
    """

    statistics = (
        df.groupby(SERIES_COL)[VALUE_COL]
        .agg(
            observations="count",
            minimum="min",
            maximum="max",
            mean="mean",
            std="std"
        )
        .reset_index()
    )

    return statistics


# ============================================================
# 10. DATA COVERAGE PER SERIES
# ============================================================

def get_series_coverage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine the start date, end date and
    number of observations for each series.
    """

    coverage = (
        df.groupby(SERIES_COL)
        .agg(
            start_date=(
                TIMESTAMP_COL,
                "min"
            ),
            end_date=(
                TIMESTAMP_COL,
                "max"
            ),
            observations=(
                VALUE_COL,
                "count"
            )
        )
        .reset_index()
    )

    return coverage


# ============================================================
# 11. SEASONALITY DETECTION
# ============================================================

def detect_seasonality(
    df: pd.DataFrame,
    periods: Optional[list] = None,
    max_series: Optional[int] = None,
) -> pd.DataFrame:
    """
    Detect potential seasonal patterns using
    autocorrelation.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataset.

    periods : list
        Candidate seasonal periods.

        For 15-minute electricity data:

            4   -> hourly
            96  -> daily
            672 -> weekly

    max_series : int or None
        Number of series to analyze.
        None = analyze all series.

    Returns
    -------
    pd.DataFrame
        Autocorrelation for each series and period.
    """

    if periods is None:

        # LD2011_2014 is commonly 15-minute data.
        #
        # 4 observations = 1 hour
        # 96 observations = 1 day
        # 672 observations = 1 week

        periods = [
            4,
            96,
            672
        ]

    results = []

    series_ids = df[
        SERIES_COL
    ].unique()

    if max_series is not None:
        series_ids = series_ids[
            :max_series
        ]

    for series_id in series_ids:

        group = (
            df[
                df[SERIES_COL]
                == series_id
            ]
            .sort_values(TIMESTAMP_COL)
        )

        values = (
            group[VALUE_COL]
            .dropna()
            .values
        )

        if len(values) < 10:
            continue

        for period in periods:

            if period >= len(values):
                continue

            x = values[:-period]
            y = values[period:]

            if (
                np.std(x) == 0
                or np.std(y) == 0
            ):
                correlation = np.nan

            else:
                correlation = np.corrcoef(
                    x,
                    y
                )[0, 1]

            results.append({
                SERIES_COL: series_id,
                "period": period,
                "autocorrelation": correlation
            })

    return pd.DataFrame(
        results
    )


# ============================================================
# 12. SEASONALITY SUMMARY
# ============================================================

def summarize_seasonality(
    seasonality_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the strongest seasonal period
    for each series.
    """

    if seasonality_df.empty:
        return seasonality_df

    summary = (
        seasonality_df
        .sort_values(
            "autocorrelation",
            ascending=False
        )
        .groupby(SERIES_COL)
        .first()
        .reset_index()
    )

    return summary


# ============================================================
# 13. COMPLETE DATASET REPORT
# ============================================================

def dataset_report(
    df: pd.DataFrame,
) -> dict:
    """
    Generate complete dataset inspection report.
    """

    basic = get_basic_statistics(
        df
    )

    missing = check_missing_values(
        df
    )

    frequency = detect_frequency(
        df
    )

    report = {
        **basic,
        "frequency": frequency,
        **missing,
    }

    return report


# ============================================================
# 14. PRINT DATASET REPORT
# ============================================================

def print_dataset_report(
    df: pd.DataFrame,
):
    """
    Print a readable Day 3 dataset report.
    """

    report = dataset_report(
        df
    )

    print()
    print("=" * 70)
    print("DAY 3 — DATASET INSPECTION")
    print("=" * 70)

    print(
        f"Observations       : "
        f"{report['observations']:,}"
    )

    print(
        f"Number of series   : "
        f"{report['number_of_series']:,}"
    )

    print(
        f"Frequency          : "
        f"{report['frequency']}"
    )

    print(
        f"Start date         : "
        f"{report['start_date']}"
    )

    print(
        f"End date           : "
        f"{report['end_date']}"
    )

    print(
        f"Minimum value      : "
        f"{report['minimum']}"
    )

    print(
        f"Maximum value      : "
        f"{report['maximum']}"
    )

    print()
    print("Missing Values")
    print("-" * 70)

    for column, count in (
        report["missing_by_column"]
        .items()
    ):

        print(
            f"{column:<20}: "
            f"{count:,}"
        )

    print(
        f"\nDuplicate "
        f"timestamp-series pairs : "
        f"{report['duplicate_timestamp_series']:,}"
    )

    print("=" * 70)


# ============================================================
# 15. ONE-CALL DATASET INSPECTION
# ============================================================

def inspect_dataset(
    file_path: str = DEFAULT_DATA_PATH,
):
    """
    Complete Day 3 pipeline.

    Loads the raw dataset, converts it to long format,
    generates the report and returns both dataframe
    and report.
    """

    df = load_dataset(
        file_path
    )

    report = dataset_report(
        df
    )

    print_dataset_report(
        df
    )

    return df, report


# ============================================================
# 16. TEST FROM COMMAND LINE
# ============================================================

if __name__ == "__main__":

    df, report = inspect_dataset(
        DEFAULT_DATA_PATH
    )

    print("\nFirst 10 rows:")
    print(df.head(10))