"""
src/evaluation.py

Reusable evaluation utilities for the
Chronos-2 / Darts forecasting project.

Responsibilities
----------------
- Calculate forecasting metrics
- Evaluate forecasts
- Compare models
- Load saved experiment results
- Save metrics and forecasts
- Prepare results for visualization

The actual experiments and interpretation should be
performed in:

    notebooks/04_evaluation.ipynb
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.metrics import mae, rmse


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"

DEFAULT_METRICS_FILE = DEFAULT_RESULTS_DIR / "metrics.csv"
DEFAULT_FORECASTS_FILE = DEFAULT_RESULTS_DIR / "forecasts.csv"


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    actual: TimeSeries,
    forecast: TimeSeries,
) -> dict:
    """
    Calculate MAE and RMSE for a forecast.

    Parameters
    ----------
    actual : TimeSeries
        Actual observations.

    forecast : TimeSeries
        Predicted observations.

    Returns
    -------
    dict
        Dictionary containing MAE and RMSE.
    """

    if len(actual) != len(forecast):
        raise ValueError(
            "Actual and forecast must have the same length."
        )

    return {
        "MAE": mae(actual, forecast),
        "RMSE": rmse(actual, forecast),
    }


# ============================================================
# EVALUATE ONE FORECAST
# ============================================================

def evaluate_forecast(
    actual: TimeSeries,
    forecast: TimeSeries,
    model_name: str,
    series_id: Optional[str] = None,
    horizon: Optional[int] = None,
) -> dict:
    """
    Evaluate one forecasting result.

    Returns a single dictionary that can later be
    converted into a results DataFrame.
    """

    metrics = calculate_metrics(
        actual=actual,
        forecast=forecast,
    )

    if horizon is None:
        horizon = len(forecast)

    result = {
        "Model": model_name,
        "Series_ID": series_id,
        "Horizon": horizon,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
    }

    return result


# ============================================================
# EVALUATE MULTIPLE FORECASTS
# ============================================================

def evaluate_multiple_forecasts(
    experiments: list,
) -> pd.DataFrame:
    """
    Evaluate multiple forecast experiments.

    Parameters
    ----------
    experiments : list of dict

        Each dictionary should contain:

            {
                "model_name": str,
                "actual": TimeSeries,
                "forecast": TimeSeries,
                "series_id": optional str,
                "horizon": optional int
            }

    Returns
    -------
    pd.DataFrame
        Model evaluation results.
    """

    results = []

    for experiment in experiments:

        result = evaluate_forecast(
            actual=experiment["actual"],
            forecast=experiment["forecast"],
            model_name=experiment["model_name"],
            series_id=experiment.get("series_id"),
            horizon=experiment.get("horizon"),
        )

        results.append(result)

    return pd.DataFrame(results)


# ============================================================
# SORT RESULTS
# ============================================================

def sort_results(
    metrics: pd.DataFrame,
    metric: str = "RMSE",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Sort model results according to a metric.

    Lower MAE/RMSE is better.
    """

    if metric not in metrics.columns:
        raise ValueError(
            f"Metric '{metric}' not found in results."
        )

    return (
        metrics
        .sort_values(
            by=metric,
            ascending=ascending,
        )
        .reset_index(drop=True)
    )


# ============================================================
# BEST MODEL
# ============================================================

def get_best_model(
    metrics: pd.DataFrame,
    metric: str = "RMSE",
) -> pd.Series:
    """
    Return the best-performing model according
    to the selected metric.

    Lower error is better.
    """

    sorted_metrics = sort_results(
        metrics,
        metric=metric,
        ascending=True,
    )

    return sorted_metrics.iloc[0]


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics: pd.DataFrame,
    path: Union[str, Path] = DEFAULT_METRICS_FILE,
) -> Path:
    """
    Save evaluation metrics to CSV.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================
# LOAD METRICS
# ============================================================

def load_metrics(
    path: Union[str, Path] = DEFAULT_METRICS_FILE,
) -> pd.DataFrame:
    """
    Load previously saved model metrics.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    return pd.read_csv(path)


# ============================================================
# SAVE FORECAST
# ============================================================

def save_forecast(
    forecast: TimeSeries,
    model_name: str,
    series_id: str,
    horizon: int,
    path: Union[str, Path] = DEFAULT_FORECASTS_FILE,
) -> Path:
    """
    Append one forecast to forecasts.csv.

    Stored columns:

        timestamp
        series_id
        model
        horizon
        forecast
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_df = forecast.to_dataframe().reset_index()

    timestamp_column = forecast_df.columns[0]
    value_column = forecast_df.columns[1]

    forecast_df = forecast_df.rename(
        columns={
            timestamp_column: "timestamp",
            value_column: "forecast",
        }
    )

    forecast_df["series_id"] = series_id
    forecast_df["model"] = model_name
    forecast_df["horizon"] = horizon

    forecast_df = forecast_df[
        [
            "timestamp",
            "series_id",
            "model",
            "horizon",
            "forecast",
        ]
    ]

    if path.exists():

        forecast_df.to_csv(
            path,
            mode="a",
            header=False,
            index=False,
        )

    else:

        forecast_df.to_csv(
            path,
            index=False,
        )

    return path


# ============================================================
# LOAD FORECASTS
# ============================================================

def load_forecasts(
    path: Union[str, Path] = DEFAULT_FORECASTS_FILE,
) -> pd.DataFrame:
    """
    Load previously saved forecasts.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {path}"
        )

    forecasts = pd.read_csv(
        path,
        parse_dates=["timestamp"],
    )

    return forecasts


# ============================================================
# COMPARE MODELS
# ============================================================

def compare_models(
    metrics: pd.DataFrame,
    metric: str = "RMSE",
) -> pd.DataFrame:
    """
    Create a model comparison table.

    If multiple experiments exist for the same model,
    average the metrics by model.
    """

    if metric not in metrics.columns:
        raise ValueError(
            f"Metric '{metric}' not found."
        )

    comparison = (
        metrics
        .groupby("Model", as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
        )
    )

    comparison = comparison.sort_values(
        by=metric,
        ascending=True,
    )

    return comparison.reset_index(
        drop=True
    )


# ============================================================
# COMPARE BY HORIZON
# ============================================================

def compare_by_horizon(
    metrics: pd.DataFrame,
    metric: str = "RMSE",
) -> pd.DataFrame:
    """
    Compare model performance separately for
    each forecasting horizon.
    """

    if "Horizon" not in metrics.columns:
        raise ValueError(
            "Horizon column not found."
        )

    result = (
        metrics
        .groupby(
            ["Horizon", "Model"],
            as_index=False,
        )
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
        )
        .sort_values(
            ["Horizon", metric]
        )
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# METRIC PLOT
# ============================================================

def plot_metric_comparison(
    metrics: pd.DataFrame,
    metric: str = "RMSE",
    title: Optional[str] = None,
    figsize=(10, 6),
):
    """
    Plot model performance using the selected metric.
    """

    comparison = compare_models(
        metrics,
        metric=metric,
    )

    plt.figure(figsize=figsize)

    plt.bar(
        comparison["Model"],
        comparison[metric],
    )

    if title is None:
        title = f"Model Comparison — {metric}"

    plt.title(title)

    plt.xlabel("Model")
    plt.ylabel(metric)

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# HORIZON PERFORMANCE PLOT
# ============================================================

def plot_horizon_comparison(
    metrics: pd.DataFrame,
    metric: str = "RMSE",
    figsize=(10, 6),
):
    """
    Plot model performance across forecasting horizons.
    """

    comparison = compare_by_horizon(
        metrics,
        metric=metric,
    )

    plt.figure(figsize=figsize)

    for model_name in comparison["Model"].unique():

        model_data = comparison[
            comparison["Model"] == model_name
        ]

        plt.plot(
            model_data["Horizon"],
            model_data[metric],
            marker="o",
            label=model_name,
        )

    plt.title(
        f"{metric} Across Forecast Horizons"
    )

    plt.xlabel(
        "Forecast Horizon (steps)"
    )

    plt.ylabel(metric)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# FORECAST VISUALIZATION
# ============================================================

def plot_forecasts(
    actual: TimeSeries,
    forecasts: dict,
    title: str = "Forecast Comparison",
    figsize=(16, 6),
):
    """
    Plot actual values together with multiple forecasts.

    Parameters
    ----------
    actual : TimeSeries
        Actual observations.

    forecasts : dict
        Dictionary:

            {
                "Model A": forecast_a,
                "Model B": forecast_b,
            }
    """

    plt.figure(figsize=figsize)

    actual.plot(
        label="Actual"
    )

    for model_name, forecast in forecasts.items():

        forecast.plot(
            label=model_name
        )

    plt.title(title)

    plt.xlabel("Time")
    plt.ylabel("Electricity Consumption")

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# ERROR SUMMARY
# ============================================================

def get_error_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate descriptive statistics for model errors.
    """

    return (
        metrics[
            ["MAE", "RMSE"]
        ]
        .describe()
    )
#############################
# CALCULATE IMPROVEMENT
#############################
def calculate_improvement(
    metrics: pd.DataFrame,
    baseline_model: str,
    metric: str = "RMSE",
) -> pd.DataFrame:
    """
    Calculate percentage improvement over a baseline model.
    """

    baseline = (
        metrics[
            metrics["Model"] == baseline_model
        ][["Horizon", metric]]
        .rename(
            columns={
                metric: f"Baseline_{metric}"
            }
        )
    )

    result = metrics.merge(
        baseline,
        on="Horizon",
        how="left",
    )

    result["Improvement_%"] = (
        (
            result[f"Baseline_{metric}"]
            - result[metric]
        )
        / result[f"Baseline_{metric}"]
        * 100
    )

    return result
###########################
# SAVE ACTUALS
############################
def save_actuals(actual, series_id, horizon, path):
    """
    Save ground-truth actual values for a given series and horizon.

    Deduplicates on (timestamp, series_id, horizon), so calling this
    once per experiment — even across multiple models or notebooks —
    never creates duplicate rows for values that are already saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    actual_df = actual.to_dataframe().reset_index()
    actual_df = actual_df.rename(columns={actual_df.columns[0]: "timestamp"})
    actual_df = actual_df.rename(columns={actual_df.columns[-1]: "value"})

    actual_df["series_id"] = series_id
    actual_df["horizon"] = horizon
    actual_df = actual_df[["timestamp", "series_id", "horizon", "value"]]

    if path.exists():
        existing = pd.read_csv(path, parse_dates=["timestamp"])
        combined = pd.concat([existing, actual_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["timestamp", "series_id", "horizon"]
        )
    else:
        combined = actual_df

    combined.to_csv(path, index=False)


def load_actuals(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Actuals file not found: {path}")
    return pd.read_csv(path, parse_dates=["timestamp"])