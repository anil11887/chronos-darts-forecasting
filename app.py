"""
app.py

Chronos-2 Electricity Forecasting — Streamlit dashboard.

Architecture:
    This app is a PRESENTATION layer only. It never executes notebooks.
    It reads pre-computed results from results/*.csv (Demo Mode), and
    optionally calls into src/ modules directly for Live Forecast Mode
    (never the notebooks themselves).

Run locally with:
    streamlit run app.py
"""

import gzip
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "LD2011_2014.txt"
DATA_GZ_PATH = PROJECT_ROOT / "data" / "raw" / "LD2011_2014.txt.gz"

# Uploaded to the release as a gzip-compressed asset to cut download
# time on Streamlit Cloud's cold starts (the raw file is ~678 MB;
# gzip brings that down substantially since it's a repetitive CSV).
DATA_URL = (
    "https://github.com/anil11887/chronos-darts-forecasting/"
    "releases/latest/download/LD2011_2014.txt.gz"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import load_dataset, load_raw_dataset  # noqa: E402
from src.advanced import get_series_columns  # noqa: E402


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Chronos-2 Electricity Forecasting",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# CACHED LOADERS
#
# Demo Mode data: cheap CSV reads, cached so repeat navigation
# between sections doesn't re-read from disk.
# ============================================================

@st.cache_data
def try_load_csv(filename, parse_dates=None):
    """
    Load a results CSV if it exists. Returns None (not an
    exception) when the file is missing, so sections can
    degrade gracefully instead of crashing the whole app.
    """
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


@st.cache_resource
def download_dataset_if_needed():
    """
    Download and decompress LD2011_2014.txt when it is not available
    locally.

    Local:
        Uses data/raw/LD2011_2014.txt directly.

    Streamlit Cloud:
        Downloads the gzip-compressed dataset from the GitHub Release
        asset, then decompresses it to data/raw/LD2011_2014.txt.
    """
    if DATA_PATH.exists():
        return DATA_PATH

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    st.info("Downloading LD2011_2014 dataset...")

    try:
        with requests.get(
            DATA_URL,
            stream=True,
            timeout=300,
        ) as response:

            response.raise_for_status()

            total_size = int(
                response.headers.get("content-length", 0)
            )

            downloaded = 0

            with open(DATA_GZ_PATH, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

                        if total_size:
                            progress = min(
                                downloaded / total_size,
                                1.0,
                            )
                            st.progress(
                                progress,
                                text=(
                                    f"Downloading dataset: "
                                    f"{progress:.0%}"
                                ),
                            )

        st.info("Decompressing dataset...")

        with gzip.open(DATA_GZ_PATH, "rb") as f_in:
            with open(DATA_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        DATA_GZ_PATH.unlink()

        return DATA_PATH

    except Exception as exc:
        if DATA_PATH.exists():
            DATA_PATH.unlink()
        if DATA_GZ_PATH.exists():
            DATA_GZ_PATH.unlink()

        raise RuntimeError(
            "Unable to download or decompress LD2011_2014.txt. "
            f"Download URL: {DATA_URL}\n\n"
            f"Original error: {exc}"
        ) from exc


@st.cache_data
def load_raw_wide_df():
    """Load the LD2011_2014 dataset as a wide dataframe."""
    dataset_path = download_dataset_if_needed()
    return load_raw_dataset(str(dataset_path))


@st.cache_data
def load_long_df():
    """Load the LD2011_2014 dataset as a long dataframe."""
    dataset_path = download_dataset_if_needed()
    return load_dataset(str(dataset_path))


def build_chronos_model(input_chunk_length=512, output_chunk_length=96):
    """
    Build a fresh, untrained Chronos-2 model instance.

    Deliberately NOT cached with @st.cache_resource: fit() mutates the
    model's internal state to match whatever covariates were used on
    that call. If we reused a cached instance across clicks with
    different covariate settings (e.g. "Time covariates" then "None"
    at the same horizon), the second .fit() call would try to refit
    a network already wired for a different input dimensionality and
    raise a ValueError. Always build fresh; the model architecture
    itself is cheap to construct — the actual cost is fit()/predict(),
    which we can't avoid re-running per click anyway since the series,
    horizon, or covariates may differ each time.
    """
    from darts.models import Chronos2Model

    return Chronos2Model(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("⚡ Navigation")

SECTIONS = [
    "🏠 Overview",
    "📊 Data Explorer",
    "🔮 Forecast",
    "📈 Model Comparison",
    "🔢 Multi-Series Experiment",
    "🧩 Covariate Experiment",
    "⏱️ Forecast Horizon Analysis",
]

section = st.sidebar.radio("Go to", SECTIONS)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data and results are pre-computed by the project notebooks "
    "(`notebooks/01`–`06`). This app reads from `results/` and "
    "never re-runs training itself, except in Live Forecast Mode."
)


# ============================================================
# SECTION: OVERVIEW
# ============================================================

if section == "🏠 Overview":

    st.title("⚡ Chronos-2 Electricity Forecasting")
    st.caption("Foundation-model forecasting for electricity demand")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", "LD2011_2014")
    col2.metric("Series", "370")
    col3.metric("Frequency", "15 min")

    col4, col5, col6 = st.columns(3)
    col4.metric("Historical Period", "2011–2014")

    metrics_df = try_load_csv("metrics.csv")
    if metrics_df is not None:
        col5.metric("Models Evaluated", metrics_df["Model"].nunique())
        col6.metric("Best 96-step RMSE", f"{metrics_df[metrics_df['Horizon'] == 96]['RMSE'].min():.3f}")

    st.markdown(
        """
        ### About this project

        This dashboard summarizes an electricity demand forecasting study using
        [Darts](https://unit8co.github.io/darts/) and **Chronos-2**, a foundation
        time-series model, benchmarked against classical baselines (Naive Seasonal,
        ARIMA) on the UCI **LD2011_2014** electricity load dataset.

        Use the sidebar to explore:
        - Raw series behavior (**Data Explorer**)
        - Actual-vs-forecast comparisons (**Forecast**)
        - Baseline vs Chronos-2 performance (**Model Comparison**)
        - How performance scales with more series (**Multi-Series Experiment**)
        - Whether deterministic time covariates help (**Covariate Experiment**)
        - How performance degrades over longer horizons (**Forecast Horizon Analysis**)
        """
    )


# ============================================================
# SECTION: DATA EXPLORER
# ============================================================

elif section == "📊 Data Explorer":

    st.title("📊 Data Explorer")

    with st.spinner("Loading dataset..."):
        raw_df = load_raw_wide_df()

    all_series = get_series_columns(raw_df)

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_series = st.selectbox("Select Series", all_series, index=0)
        show_historical = st.checkbox("Historical data", value=True)
        show_daily = st.checkbox("Daily pattern (avg by hour)", value=False)
        show_weekly = st.checkbox("Weekly pattern (avg by day)", value=False)

    series_data = raw_df[["timestamp", selected_series]].copy()
    series_data["timestamp"] = pd.to_datetime(series_data["timestamp"])

    with col2:
        if show_historical:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series_data["timestamp"], series_data[selected_series], linewidth=0.8)
            ax.set_title(f"{selected_series} — Historical Consumption")
            ax.set_xlabel("Time")
            ax.set_ylabel("Electricity Consumption")
            st.pyplot(fig)

    if show_daily:
        hourly = series_data.copy()
        hourly["hour"] = hourly["timestamp"].dt.hour
        hourly_avg = hourly.groupby("hour")[selected_series].mean()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(hourly_avg.index, hourly_avg.values, marker="o")
        ax.set_title(f"{selected_series} — Average Daily Pattern")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg Consumption")
        st.pyplot(fig)

    if show_weekly:
        weekly = series_data.copy()
        weekly["dow"] = weekly["timestamp"].dt.dayofweek
        weekly_avg = weekly.groupby("dow")[selected_series].mean()

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            weekly_avg.values,
        )
        ax.set_title(f"{selected_series} — Average Weekly Pattern")
        ax.set_ylabel("Avg Consumption")
        st.pyplot(fig)


# ============================================================
# SECTION: FORECAST
# ============================================================

elif section == "🔮 Forecast":

    st.title("🔮 Forecast")

    mode = st.radio("Mode", ["Demo (precomputed)", "Live (runs Chronos-2 now)"], horizontal=True)

    forecasts_df = try_load_csv("forecasts.csv", parse_dates=["timestamp"])
    actuals_df = try_load_csv("actuals.csv", parse_dates=["timestamp"])

    if mode == "Demo (precomputed)":

        if forecasts_df is None or actuals_df is None:
            st.warning(
                "No precomputed forecasts found. Run notebooks 02–04 first to "
                "generate `results/forecasts.csv` and `results/actuals.csv`."
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                series_options = sorted(forecasts_df["series_id"].unique())
                selected_series = st.selectbox(
                    "Series (precomputed)", series_options,
                )
                if len(series_options) < 370:
                    st.caption(
                        f"Only {len(series_options)} series have precomputed demo "
                        "forecasts. Switch to **Live** mode above to forecast any "
                        "of the other series on demand."
                    )
            with col2:
                horizon_options = sorted(forecasts_df["horizon"].unique())
                horizon_labels = {96: "24 hours", 192: "48 hours", 672: "7 days"}
                selected_horizon = st.selectbox(
                    "Forecast Horizon",
                    horizon_options,
                    format_func=lambda h: horizon_labels.get(h, f"{h} steps"),
                )

            f_slice = forecasts_df[
                (forecasts_df["series_id"] == selected_series)
                & (forecasts_df["horizon"] == selected_horizon)
            ].sort_values("timestamp")

            a_slice = actuals_df[
                (actuals_df["series_id"] == selected_series)
                & (actuals_df["horizon"] == selected_horizon)
            ].sort_values("timestamp")

            if f_slice.empty or a_slice.empty:
                st.info("No data available for this series/horizon combination yet.")
            else:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(
                    a_slice["timestamp"], a_slice["value"],
                    label="Actual", color="black", linewidth=2,
                )
                for model_name in f_slice["model_name"].unique():
                    model_data = f_slice[f_slice["model_name"] == model_name].sort_values("timestamp")
                    ax.plot(
                        model_data["timestamp"], model_data["value"],
                        label=model_name, linestyle="--",
                    )
                ax.set_title(f"Actual vs Forecast — {selected_series}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Electricity Consumption")
                ax.legend()
                st.pyplot(fig)

                metrics_df = try_load_csv("metrics.csv")
                if metrics_df is not None:
                    row = metrics_df[
                        (metrics_df["Series_ID"] == selected_series)
                        & (metrics_df["Horizon"] == selected_horizon)
                        & (metrics_df["Model"] == "Chronos-2")
                    ]
                    if not row.empty:
                        c1, c2 = st.columns(2)
                        c1.metric("MAE", f"{row['MAE'].iloc[0]:.4f}")
                        c2.metric("RMSE", f"{row['RMSE'].iloc[0]:.4f}")

    else:  # Live Forecast Mode
        st.info(
            "Live mode runs real forecasts on the spot. Naive Seasonal is instant, "
            "ARIMA can take a while on a long series, and Chronos-2 is slow on the "
            "first run of a session (model loading) and slower on machines without a GPU."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            raw_df = load_raw_wide_df()
            all_series = get_series_columns(raw_df)
            live_series = st.selectbox("Series", all_series, key="live_series")
        with col2:
            live_horizon = st.selectbox(
                "Horizon",
                [96, 192, 672],
                format_func=lambda h: {96: "24 hours", 192: "48 hours", 672: "7 days"}[h],
            )
        with col3:
            use_covariates = st.selectbox("Covariates", ["None", "Time covariates"])

        if st.button("Generate Forecast"):
            with st.spinner("Loading model and forecasting — this may take a while..."):
                from darts import TimeSeries
                from darts.metrics import mae, rmse
                from src.advanced import create_future_cyclic_covariates

                series_df = raw_df[["timestamp", live_series]].copy()
                target = TimeSeries.from_dataframe(
                    series_df, time_col="timestamp", value_cols=live_series,
                )
                train, test = target.split_before(0.8)
                actual = test[:live_horizon]

                chronos_model = build_chronos_model(
                    input_chunk_length=512,
                    output_chunk_length=live_horizon,
                )

                fit_kwargs = {}
                predict_kwargs = {"n": live_horizon, "series": train}

                if use_covariates == "Time covariates":
                    covariates = create_future_cyclic_covariates(
                        target, forecast_horizon=live_horizon,
                    )
                    train_cov = covariates.slice_intersect(train)
                    fit_kwargs["future_covariates"] = train_cov
                    predict_kwargs["future_covariates"] = covariates

                chronos_model.fit(train, **fit_kwargs)
                forecast = chronos_model.predict(**predict_kwargs)

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(
                    actual.time_index, actual.values().flatten(),
                    label="Actual", color="black", linewidth=2,
                )
                ax.plot(
                    forecast.time_index, forecast.values().flatten(),
                    label="Chronos-2", linestyle="--",
                )
                ax.set_title(f"Live Forecast — {live_series}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Electricity Consumption")
                ax.legend()
                st.pyplot(fig)

                c1, c2 = st.columns(2)
                c1.metric("MAE", f"{mae(actual, forecast):.4f}")
                c2.metric("RMSE", f"{rmse(actual, forecast):.4f}")


# ============================================================
# SECTION: MODEL COMPARISON
# ============================================================

elif section == "📈 Model Comparison":

    st.title("📈 Model Comparison")

    metrics_df = try_load_csv("metrics.csv")

    if metrics_df is None:
        st.warning("No metrics found. Run notebooks 02–03 to generate `results/metrics.csv`.")
    else:
        horizon_options = sorted(metrics_df["Horizon"].unique())
        horizon_labels = {96: "24 hours", 192: "48 hours", 672: "7 days"}
        selected_horizon = st.selectbox(
            "Horizon",
            horizon_options,
            format_func=lambda h: horizon_labels.get(h, f"{h} steps"),
        )

        subset = metrics_df[metrics_df["Horizon"] == selected_horizon].sort_values("RMSE")

        st.subheader(f"Model Comparison — {horizon_labels.get(selected_horizon, selected_horizon)} Horizon")

        fig, ax = plt.subplots(figsize=(8, 0.6 * len(subset) + 1))
        ax.barh(subset["Model"], subset["RMSE"])
        ax.set_xlabel("RMSE (lower is better)")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.dataframe(
            subset[["Model", "Horizon", "MAE", "RMSE"]].reset_index(drop=True),
            use_container_width=True,
        )

        best_model = subset.iloc[0]["Model"]
        st.caption(
            f"Note: the best-performing model at this horizon is **{best_model}** — "
            "Chronos-2 does not automatically win against classical baselines on "
            "every horizon, which is itself a meaningful research finding."
        )


# ============================================================
# SECTION: MULTI-SERIES EXPERIMENT
# ============================================================

elif section == "🔢 Multi-Series Experiment":

    st.title("🔢 Multi-Series Experiment")

    scale_df = try_load_csv("experiment3_scale_metrics.csv")

    if scale_df is None:
        st.warning(
            "No multi-series results found. Run notebook 06 "
            "(`experiment3_scale_metrics.csv`) first."
        )
    else:
        summary = (
            scale_df.groupby("N_Series")[["MAE", "RMSE"]]
            .mean()
            .reset_index()
            .sort_values("N_Series")
        )

        st.subheader("RMSE by Number of Series")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(summary["N_Series"].astype(str), summary["RMSE"], marker="o")
        ax.set_xlabel("Number of Series")
        ax.set_ylabel("Mean RMSE")
        st.pyplot(fig)

        st.dataframe(summary, use_container_width=True)

        st.subheader("Per-Series Performance")

        available_n = sorted(scale_df["N_Series"].unique())
        selected_n = st.selectbox("Number of series in this run", available_n, index=len(available_n) - 1)

        sort_by = st.selectbox("Sort by", ["RMSE", "MAE"])

        per_series = (
            scale_df[scale_df["N_Series"] == selected_n]
            .sort_values(sort_by)
            .reset_index(drop=True)
        )

        st.dataframe(per_series, use_container_width=True)


# ============================================================
# SECTION: COVARIATE EXPERIMENT
# ============================================================

elif section == "🧩 Covariate Experiment":

    st.title("🧩 Covariate Experiment")

    cov_df = try_load_csv("covariate_ablation_metrics.csv")

    if cov_df is None:
        st.warning(
            "No covariate ablation results found. Run notebook 06 "
            "(`covariate_ablation_metrics.csv`) first."
        )
    else:
        st.subheader("Chronos-2 Covariate Ablation")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(cov_df["Experiment"], cov_df["RMSE"])
        ax.set_xlabel("RMSE (lower is better)")
        st.pyplot(fig)

        st.dataframe(cov_df, use_container_width=True)

        # The B2 row already carries its own improvement vs B1 in
        # Improvement_vs_B1_% (B1 itself is always 0%, since it's
        # the baseline it's compared against).
        improvement_row = cov_df[cov_df["Improvement_vs_B1_%"] != 0]
        if not improvement_row.empty:
            improvement = improvement_row["Improvement_vs_B1_%"].iloc[0]
            st.metric("RMSE Improvement from Covariates", f"{improvement:+.2f}%")

        st.markdown(
            """
            **Covariates used:**
            - ✓ Hour (cyclic sin/cos encoding)
            - ✓ Day of week (cyclic sin/cos encoding)
            - ✓ Month (cyclic sin/cos encoding)

            These are deterministic calendar features derived purely from the
            timestamp — no external data (e.g. temperature, humidity) is used,
            since the LD2011_2014 dataset does not provide those variables.
            """
        )


# ============================================================
# SECTION: FORECAST HORIZON ANALYSIS
# ============================================================

elif section == "⏱️ Forecast Horizon Analysis":

    st.title("⏱️ Forecast Horizon Analysis")

    horizon_df = try_load_csv("experiment5_horizon_covariate_matrix.csv")

    if horizon_df is None:
        st.warning(
            "No horizon analysis results found. Run notebook 06 "
            "(`experiment5_horizon_covariate_matrix.csv`) first."
        )
    else:
        st.subheader("Chronos-2 RMSE Across Forecast Horizons")

        fig, ax = plt.subplots(figsize=(8, 4))
        for covariate_label in horizon_df["Covariates"].unique():
            subset = horizon_df[horizon_df["Covariates"] == covariate_label].sort_values("Horizon_Steps")
            ax.plot(subset["Horizon_Label"], subset["RMSE"], marker="o", label=covariate_label)
        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel("RMSE")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(horizon_df, use_container_width=True)