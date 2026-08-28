"""
src/visualization.py

Reusable visualization utilities for the
Chronos-2 electricity forecasting project.

This module contains plotting functions only.

No model training.
No model fitting.
No forecasting.

The functions are intended to be used by:

    notebooks/05_visualization.ipynb
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_FIGURES_DIR = Path("results") / "figures"


# ============================================================
# INTERNAL HELPER
# ============================================================

def _prepare_figures_directory(
    figures_dir: Path,
) -> Path:
    """
    Create the figures directory if it does not exist.
    """

    figures_dir = Path(figures_dir)

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return figures_dir


# ============================================================
# FIGURE 1 — HISTORICAL DATA
# ============================================================

def plot_historical_data(
    series: TimeSeries,
    series_id: Optional[str] = None,
    title: Optional[str] = None,
    figsize=(16, 6),
    save_path: Optional[str] = None,
):
    """
    Plot the complete historical time series.

    Parameters
    ----------
    series : TimeSeries
        Darts time series.

    series_id : str, optional
        Identifier of the electricity series.

    title : str, optional
        Custom figure title.

    figsize : tuple
        Figure dimensions.

    save_path : str, optional
        Path to save the figure.
    """

    plt.figure(figsize=figsize)

    series.plot(
        label="Historical Data"
    )

    if title is None:

        if series_id is not None:
            title = (
                f"Historical Electricity "
                f"Consumption — {series_id}"
            )
        else:
            title = "Historical Electricity Consumption"

    plt.title(title)

    plt.xlabel("Time")
    plt.ylabel("Electricity Consumption")

    plt.legend()

    plt.tight_layout()

    if save_path is not None:

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# FIGURE 2 — ACTUAL VS CHRONOS
# ============================================================

def plot_actual_vs_chronos(
    actual: TimeSeries,
    forecast: TimeSeries,
    horizon_label: str = "",
    figsize=(16, 6),
    save_path: Optional[str] = None,
):
    """
    Plot actual observations against Chronos-2 forecast.
    """

    plt.figure(figsize=figsize)

    actual.plot(
        label="Actual"
    )

    forecast.plot(
        label="Chronos-2"
    )

    title = "Actual vs Chronos-2 Forecast"

    if horizon_label:
        title += f" — {horizon_label}"

    plt.title(title)

    plt.xlabel("Time")
    plt.ylabel("Electricity Consumption")

    plt.legend()

    plt.tight_layout()

    if save_path is not None:

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# FIGURE 2B — ACTUAL VS MULTIPLE MODELS
# ============================================================

def plot_actual_vs_models(
    actual: TimeSeries,
    forecasts: dict,
    title: str = "Actual vs Forecasts",
    figsize=(16, 6),
    save_path: Optional[str] = None,
):
    """
    Plot actual values against forecasts from
    multiple models.

    Parameters
    ----------
    actual : TimeSeries

    forecasts : dict
        Example:

            {
                "Naive Seasonal": naive_forecast,
                "ARIMA": arima_forecast,
                "Chronos-2": chronos_forecast,
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

    if save_path is not None:

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# FIGURE 3 — MODEL COMPARISON
# ============================================================

def plot_model_comparison(
    metrics: pd.DataFrame,
    metric: str = "RMSE",
    horizon: Optional[int] = None,
    title: Optional[str] = None,
    figsize=(10, 6),
    save_path: Optional[str] = None,
):
    """
    Compare forecasting models using MAE or RMSE.

    Parameters
    ----------
    metrics : DataFrame
        Results from evaluation.py.

    metric : str
        Either "MAE" or "RMSE".

    horizon : int, optional
        If supplied, compare only that horizon.
    """

    if metric not in ["MAE", "RMSE"]:
        raise ValueError(
            "metric must be either 'MAE' or 'RMSE'."
        )

    data = metrics.copy()

    if horizon is not None:

        data = data[
            data["Horizon"] == horizon
        ].copy()

    if data.empty:
        raise ValueError(
            "No results available for the selected horizon."
        )

    # If multiple results exist for a model,
    # average them.
    comparison = (
        data
        .groupby("Model", as_index=False)[metric]
        .mean()
        .sort_values(
            metric,
            ascending=True,
        )
    )

    plt.figure(figsize=figsize)

    plt.bar(
        comparison["Model"],
        comparison[metric],
    )

    if title is None:

        title = (
            f"Model Comparison — {metric}"
        )

        if horizon is not None:
            title += f" — {horizon} Steps"

    plt.title(title)

    plt.xlabel("Model")
    plt.ylabel(metric)

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    if save_path is not None:

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# FIGURE 4 — FORECAST HORIZON COMPARISON
# ============================================================

def plot_forecast_horizons(
    metrics: pd.DataFrame,
    metric: str = "RMSE",
    model_name: str = "Chronos-2",
    figsize=(10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot how one model performs across
    different forecasting horizons.
    """

    if metric not in ["MAE", "RMSE"]:
        raise ValueError(
            "metric must be either 'MAE' or 'RMSE'."
        )

    data = metrics[
        metrics["Model"] == model_name
    ].copy()

    if data.empty:
        raise ValueError(
            f"No results found for model: {model_name}"
        )

    data = (
        data
        .groupby("Horizon", as_index=False)[metric]
        .mean()
        .sort_values("Horizon")
    )

    plt.figure(figsize=figsize)

    plt.plot(
        data["Horizon"],
        data[metric],
        marker="o",
        label=model_name,
    )

    plt.title(
        f"{model_name} — {metric} Across Forecast Horizons"
    )

    plt.xlabel(
        "Forecast Horizon (steps)"
    )

    plt.ylabel(metric)

    plt.xticks(
        data["Horizon"]
    )

    plt.legend()

    plt.tight_layout()

    if save_path is not None:

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# FIGURE 4B — ALL MODELS ACROSS HORIZONS
# ============================================================

def plot_all_models_horizons(
    metrics: pd.DataFrame,
    metric: str = "RMSE",
    figsize=(12, 6),
    save_path: Optional[str] = None,
):
    """
    Compare all available models across
    forecasting horizons.
    """

    if metric not in ["MAE", "RMSE"]:
        raise ValueError(
            "metric must be either 'MAE' or 'RMSE'."
        )

    data = (
        metrics
        .groupby(
            ["Model", "Horizon"],
            as_index=False,
        )[metric]
        .mean()
        .sort_values("Horizon")
    )

    plt.figure(figsize=figsize)

    for model_name in data["Model"].unique():

        model_data = data[
            data["Model"] == model_name
        ]

        plt.plot(
            model_data["Horizon"],
            model_data[metric],
            marker="o",
            label=model_name,
        )

    plt.title(
        f"Model Performance Across Forecast Horizons — {metric}"
    )

    plt.xlabel(
        "Forecast Horizon (steps)"
    )

    plt.ylabel(metric)

    plt.legend()

    plt.tight_layout()

    if save_path is not None:

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# FORECAST CSV VISUALIZATION
# ============================================================

def plot_saved_forecasts(
    forecasts: pd.DataFrame,
    actuals: Optional[pd.DataFrame] = None,
    horizon: int = 96,
    figsize=(16, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot forecasts stored in forecasts.csv.

    If actuals are supplied, they are plotted first.

    Expected forecast columns:

        timestamp
        series_id
        model
        horizon
        forecast

    Expected actual columns:

        timestamp
        series_id
        horizon
        actual
    """

    forecast_data = forecasts[
        forecasts["horizon"] == horizon
    ].copy()

    if forecast_data.empty:
        raise ValueError(
            f"No forecasts found for horizon={horizon}."
        )

    plt.figure(figsize=figsize)

    # Actual values
    if actuals is not None:

        actual_data = actuals[
            actuals["horizon"] == horizon
        ].copy()

        if not actual_data.empty:

            plt.plot(
                actual_data["timestamp"],
                actual_data["actual"],
                label="Actual",
            )

    # Forecasts
    for model_name in forecast_data["model"].unique():

        model_data = forecast_data[
            forecast_data["model"] == model_name
        ]

        plt.plot(
            model_data["timestamp"],
            model_data["forecast"],
            label=model_name,
        )

    if title is None:

        title = (
            f"Forecast Comparison — "
            f"{horizon} Steps"
        )

    plt.title(title)

    plt.xlabel("Time")
    plt.ylabel("Electricity Consumption")

    plt.legend()

    plt.tight_layout()

    if save_path is not None:

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# SAVE ALL FIGURES DIRECTORY
# ============================================================

def get_figures_directory(
    results_dir: str = "results",
) -> Path:
    """
    Return and create the project's figures directory.
    """

    figures_dir = (
        Path(results_dir) / "figures"
    )

    return _prepare_figures_directory(
        figures_dir
    )