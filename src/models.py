"""
src/models.py

Reusable forecasting model utilities for the
Chronos-2 / Darts forecasting project.


    Chronological train/test splitting


    Naive Seasonal baseline


    ARIMA baseline

The actual experimentation, visualization and
evaluation should be performed in:

    notebooks/02_baselines.ipynb
"""

from typing import Optional, Tuple

from darts import TimeSeries
from darts.models import NaiveSeasonal, ARIMA


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_NAIVE_K = 24


# ============================================================
#  — CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

def chronological_train_test_split(
    series: TimeSeries,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Tuple[TimeSeries, TimeSeries]:
    """
    Split a time series chronologically.

    No random shuffling is performed.

    Parameters
    ----------
    series : TimeSeries
        Darts TimeSeries object.

    train_ratio : float
        Fraction of observations assigned to training.

        Default:
            0.80

    Returns
    -------
    train : TimeSeries
        Historical training portion.

    test : TimeSeries
        Future test portion.
    """

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    train, test = series.split_before(
        train_ratio
    )

    return train, test


# ============================================================
# SPLIT INFORMATION
# ============================================================

def get_split_information(
    train: TimeSeries,
    test: TimeSeries,
) -> dict:
    """
    Return useful information about the train/test split.
    """

    train_length = len(train)
    test_length = len(test)

    total_length = (
        train_length + test_length
    )

    return {
        "train_length": train_length,
        "test_length": test_length,
        "total_length": total_length,
        "train_percentage": (
            train_length / total_length * 100
        ),
        "test_percentage": (
            test_length / total_length * 100
        ),
        "train_start": train.start_time(),
        "train_end": train.end_time(),
        "test_start": test.start_time(),
        "test_end": test.end_time(),
    }


# ============================================================
#  — NAIVE SEASONAL MODEL
# ============================================================

def create_naive_seasonal_model(
    K: int = DEFAULT_NAIVE_K,
) -> NaiveSeasonal:
    """
    Create a Darts NaiveSeasonal model.

    Parameters
    ----------
    K : int
        Number of time steps used for the seasonal naive
        forecast.

        For your 15-minute dataset:

            K = 4     -> 1 hour
            K = 24    -> 6 hours
            K = 96    -> 1 day
            K = 672   -> 1 week

    Returns
    -------
    NaiveSeasonal
        Unfitted Darts model.
    """

    if K <= 0:
        raise ValueError(
            "K must be a positive integer."
        )

    return NaiveSeasonal(
        K=K
    )


# ============================================================
# FIT NAIVE SEASONAL
# ============================================================

def fit_naive_seasonal(
    train: TimeSeries,
    K: int = DEFAULT_NAIVE_K,
) -> NaiveSeasonal:
    """
    Create and fit a NaiveSeasonal model.
    """

    model = create_naive_seasonal_model(
        K=K
    )

    model.fit(train)

    return model


# ============================================================
# NAIVE FORECAST
# ============================================================

def forecast_naive_seasonal(
    train: TimeSeries,
    forecast_horizon: int,
    K: int = DEFAULT_NAIVE_K,
) -> TimeSeries:
    """
    Fit a NaiveSeasonal model and generate a forecast.

    Parameters
    ----------
    train : TimeSeries
        Training series.

    forecast_horizon : int
        Number of future time steps to predict.

    K : int
        Seasonal lag.

    Returns
    -------
    TimeSeries
        Forecast.
    """

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be positive."
        )

    model = fit_naive_seasonal(
        train=train,
        K=K
    )

    forecast = model.predict(
        forecast_horizon
    )

    return forecast


# ============================================================
# — ARIMA MODEL
# ============================================================

def create_arima_model(
    p: int = 12,
    d: int = 1,
    q: int = 0,
) -> ARIMA:
    """
    Create a Darts ARIMA model.

    Parameters
    ----------
    p : int
        AR order.

    d : int
        Differencing order.

    q : int
        MA order.

    Returns
    -------
    ARIMA
        Unfitted Darts ARIMA model.

    Notes
    -----
    The default order is configurable so that
    experimentation can be performed in the notebook.
    """

    if p < 0:
        raise ValueError(
            "p must be >= 0."
        )

    if d < 0:
        raise ValueError(
            "d must be >= 0."
        )

    if q < 0:
        raise ValueError(
            "q must be >= 0."
        )

    return ARIMA(
        p=p,
        d=d,
        q=q
    )


# ============================================================
# FIT ARIMA
# ============================================================

def fit_arima(
    train: TimeSeries,
    p: int = 12,
    d: int = 1,
    q: int = 0,
) -> ARIMA:
    """
    Create and fit an ARIMA model.
    """

    model = create_arima_model(
        p=p,
        d=d,
        q=q
    )

    model.fit(train)

    return model


# ============================================================
# ARIMA FORECAST
# ============================================================

def forecast_arima(
    train: TimeSeries,
    forecast_horizon: int,
    p: int = 12,
    d: int = 1,
    q: int = 0,
) -> TimeSeries:
    """
    Fit ARIMA and generate a forecast.

    Parameters
    ----------
    train : TimeSeries
        Training series.

    forecast_horizon : int
        Number of future time steps.

    p, d, q : int
        ARIMA parameters.

    Returns
    -------
    TimeSeries
        Forecast.
    """

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be positive."
        )

    model = fit_arima(
        train=train,
        p=p,
        d=d,
        q=q
    )

    forecast = model.predict(
        forecast_horizon
    )

    return forecast


# ============================================================
# GENERIC MODEL FORECAST
# ============================================================

def fit_and_predict(
    model,
    train: TimeSeries,
    forecast_horizon: int,
):
    """
    Generic helper for any Darts forecasting model.

    This follows the common Darts pattern:

        model.fit(train)
        forecast = model.predict(n)

    Parameters
    ----------
    model : Darts forecasting model
        Any compatible Darts forecasting model.

    train : TimeSeries
        Training series.

    forecast_horizon : int
        Number of future observations.

    Returns
    -------
    model
        Fitted model.

    forecast
        Forecasted TimeSeries.
    """

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be positive."
        )

    model.fit(train)

    forecast = model.predict(
        forecast_horizon
    )

    return model, forecast

# ============================================================
# — CHRONOS-2
# ============================================================

def create_chronos2_model(
    input_chunk_length: int = 512,
    output_chunk_length: int = 96,
):
    """
    Create a Darts Chronos-2 forecasting model.

    Parameters
    ----------
    input_chunk_length : int
        Number of historical time steps given to Chronos-2.

    output_chunk_length : int
        Maximum number of time steps predicted at once.

        For the LD2011_2014 dataset:

            96  -> 24 hours
            192 -> 48 hours
            672 -> 7 days

    Returns
    -------
    Chronos2Model
        Unfitted Chronos-2 model.
    """

    from darts.models import Chronos2Model

    if input_chunk_length <= 0:
        raise ValueError(
            "input_chunk_length must be positive."
        )

    if output_chunk_length <= 0:
        raise ValueError(
            "output_chunk_length must be positive."
        )

    model = Chronos2Model(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
    )

    return model


# ============================================================
# FIT CHRONOS-2
# ============================================================

def fit_chronos2(
    train,
    input_chunk_length: int = 512,
    output_chunk_length: int = 96,
):
    """
    Create and fit a Chronos-2 model.

    Parameters
    ----------
    train : TimeSeries
        Training time series.

    input_chunk_length : int
        Historical context length.

    output_chunk_length : int
        Forecast horizon supported by the model.

    Returns
    -------
    Chronos2Model
        Fitted Chronos-2 model.
    """

    model = create_chronos2_model(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
    )

    model.fit(train)

    return model


# ============================================================
# CHRONOS-2 FORECAST
# ============================================================

def forecast_chronos2(
    train,
    forecast_horizon: int,
    input_chunk_length: int = 512,
):
    """
    Fit Chronos-2 and generate a forecast.

    The output chunk length is set to the requested
    forecasting horizon.

    Parameters
    ----------
    train : TimeSeries
        Training time series.

    forecast_horizon : int
        Number of future time steps.

    input_chunk_length : int
        Historical context length.

    Returns
    -------
    model
        Fitted Chronos-2 model.

    forecast
        Chronos-2 forecast.
    """

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be positive."
        )

    model = fit_chronos2(
        train=train,
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
    )

    forecast = model.predict(
        n=forecast_horizon,
        series=train,
    )

    return model, forecast


# ============================================================
# CHRONOS-2 EXPERIMENT
# ============================================================

def run_chronos2_experiment(
    train,
    test,
    forecast_horizon: int,
    input_chunk_length: int = 512,
):
    """
    Run one complete Chronos-2 experiment.

    Steps:

        1. Create model
        2. Fit model
        3. Forecast requested horizon
        4. Compare with test data

    Notes
    -----
    The returned forecast contains only `forecast_horizon`
    future points.

    Therefore, `test` should be sliced to the same horizon
    before calculating metrics.
    """

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be positive."
        )

    if forecast_horizon > len(test):
        raise ValueError(
            "forecast_horizon cannot be larger "
            "than the available test set."
        )

    model, forecast = forecast_chronos2(
        train=train,
        forecast_horizon=forecast_horizon,
        input_chunk_length=input_chunk_length,
    )

    actual = test[:forecast_horizon]

    return model, actual, forecast


# ============================================================
#  — HISTORICAL FORECASTS / BACKTESTING
# ============================================================

def run_chronos2_historical_forecasts(
    model,
    series,
    forecast_horizon: int,
    start: float = 0.8,
    stride: int = 96,
    retrain: bool = True,
    last_points_only: bool = True,
    verbose: bool = True,
):
    """
    Generate rolling historical forecasts using Chronos-2.

    Parameters
    ----------
    model :
        Fitted Chronos-2 model.

    series : TimeSeries
        Complete time series.

    forecast_horizon : int
        Number of future steps predicted at each forecast origin.

    start : float
        Fraction of the series before the first forecast.

        Example:
            0.8 = start around 80% of the series.

    stride : int
        Number of time steps between forecast origins.

        For 15-minute data:

            4   = 1 hour
            96  = 1 day
            672 = 1 week

    retrain : bool or int
        Whether to retrain at each forecast origin.

    last_points_only : bool
        If True, return only the final point of each forecast.

    Returns
    -------
    TimeSeries or list[TimeSeries]
        Historical forecasts.
    """

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be positive."
        )

    if not 0 < start < 1:
        raise ValueError(
            "start must be between 0 and 1."
        )

    if stride <= 0:
        raise ValueError(
            "stride must be positive."
        )

    historical_forecasts = model.historical_forecasts(
        series=series,
        forecast_horizon=forecast_horizon,
        start=start,
        stride=stride,
        retrain=retrain,
        last_points_only=last_points_only,
        verbose=verbose,
    )

    return historical_forecasts