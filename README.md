# Chronos-2 Electricity Forecasting

> Electricity demand forecasting using Amazon Chronos-2 and Darts.

## 🚀 Live Demo

[Open Streamlit App](#) <!-- replace with your actual Streamlit Community Cloud URL after deployment -->

## 📌 Project Overview

This project benchmarks **Chronos-2** — a pretrained foundation model for time
series forecasting — against classical forecasting baselines (Naive Seasonal,
ARIMA) on real-world electricity consumption data.

Rather than treating Chronos-2 as an automatic win, the project is structured
as a set of **ablations**: does a foundation model actually beat simple
baselines here, does adding deterministic time covariates help, and how does
performance change as more series or longer horizons are involved? The
results are not uniformly in Chronos-2's favor — see [Results](#results) —
which is itself part of the point.

The research (notebooks) and the presentation layer (Streamlit app) are kept
deliberately separate: notebooks are the record of experimentation, `src/` is
shared, tested logic, and `app.py` only ever reads pre-computed results or
calls `src/` directly — it never executes a notebook.

## Dataset

**[LD2011_2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)**
(UCI Machine Learning Repository)

- 370 electricity consumption series (`MT_001`–`MT_370`)
- 15-minute frequency
- 2011–2014 (four years of history)
- ~140,256 timestamps per series

The raw file is semicolon-separated with comma decimals and is **not**
committed to this repository (see [`.gitignore`](.gitignore)) due to its
size (~150–190MB). See [Installation](#installation) for how to obtain it.

## Architecture

```
Raw Data (LD2011_2014.txt)
        ↓
Darts TimeSeries (wide format → per-series conversion)
        ↓
Chronological Split (80/20, train/test)
        ↓
Baseline Models (Naive Seasonal, ARIMA)
        ↓
Chronos-2 (zero-shot + with time covariates)
        ↓
Evaluation (MAE, RMSE, ablations)
        ↓
Streamlit (Demo Mode: reads results/*.csv | Live Mode: calls src/ directly)
```

Notebooks never feed the app directly — everything the app shows either
comes from a CSV in `results/`, or from a fresh call into `src/` at runtime
(Live Forecast Mode only).

## Models

| Model                  | Notes                                                      |
| ---------------------- | ---------------------------------------------------------- |
| Naive Seasonal (Daily) | Repeats the value from 96 steps (1 day) earlier            |
| ARIMA(12,1,0)          | Classical statistical baseline, fit per series             |
| Chronos-2              | Pretrained foundation model, zero-shot and with covariates |

## Forecast Horizons

| Horizon  | Steps (15-min freq) |
| -------- | ------------------- |
| 24 hours | 96                  |
| 48 hours | 192                 |
| 7 days   | 672                 |

## Advanced Experiments

### Multiple Series

Chronos-2 fit jointly across a growing number of series to see how
performance and resource cost scale:

```
10 series → 50 series → 370 series
```

See `notebooks/06_multiseries_covariates.ipynb` and
`results/experiment3_scale_metrics.csv`.

### Covariates

Deterministic, cyclic-encoded time features (safe to use for _future_
timestamps, since they require no external data):

- Hour of day
- Day of week
- Month
- Cyclic (sin/cos) encoding for all of the above, so e.g. 23:00 and 00:00
  aren't treated as numerically distant

Compared as a clean **ablation** — target-only vs. target + covariates — on
a single series (`MT_001`) before mixing in the multi-series experiment, so
the two effects aren't conflated. See
`results/covariate_ablation_metrics.csv`.

> **Note:** only calendar-derived features are used. The dataset does not
> include temperature, humidity, or other external variables, so no claims
> are made about weather-based covariates.

## Results

Full per-model, per-horizon metrics live in `results/metrics.csv`, with
per-timestep forecasts and actuals in `results/forecasts.csv` and
`results/actuals.csv`. Explore all of this interactively in the
[Streamlit app](#-live-demo) — the **Model Comparison**, **Multi-Series
Experiment**, **Covariate Experiment**, and **Forecast Horizon Analysis**
sections all render directly from these files.

Headline finding: **Chronos-2 does not automatically outperform ARIMA at
every horizon** on this dataset — see the Model Comparison section of the
app for the full breakdown. This is treated as a genuine result, not a
shortcoming of the experiment.

## Streamlit Application

`app.py` is a two-mode dashboard:

- **Demo Mode** — instant, reads pre-computed CSVs from `results/`. No model
  loading, no training, no dependency on the raw dataset being present.
- **Live Forecast Mode** — loads Chronos-2 on demand (`@st.cache_resource`
  is intentionally _not_ used for the fitted model itself, since Chronos-2
  must be re-fit per series/horizon/covariate combination — see comments in
  `app.py` for why) and runs a real forecast for any of the 370 series.

Sections: Overview, Data Explorer, Forecast, Model Comparison, Multi-Series
Experiment, Covariate Experiment, Forecast Horizon Analysis.

## Project Structure

```
chronos-2/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── app.py
│
├── data/
│   └── raw/
│       └── LD2011_2014.txt        # not committed — see Installation
│
├── src/
│   ├── data.py                    # loading, cleaning, long/wide conversion
│   ├── models.py                  # baselines, Chronos-2 fit/forecast helpers
│   ├── evaluation.py               # metrics, save/load for results CSVs
│   └── advanced.py                # multi-series + covariate utilities
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_chronos2_forecasting.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_visualization.ipynb
│   └── 06_multiseries_covariates.ipynb
│
├── results/
│   ├── metrics.csv
│   ├── forecasts.csv
│   ├── actuals.csv
│   ├── experiment3_scale_metrics.csv
│   ├── covariate_ablation_metrics.csv
│   ├── experiment5_horizon_covariate_matrix.csv
│   └── figures/
│
└── .streamlit/
    └── config.toml
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/chronos-2.git
   cd chronos-2
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**

   The raw dataset is not committed to this repo. Download `LD2011_2014.txt`
   from the
   [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
   and place it at:

   ```
   data/raw/LD2011_2014.txt
   ```

## Running Locally

**Notebooks** (research/experimentation):

```bash
jupyter notebook notebooks/
```

Run in order, `01` through `06` — later notebooks depend on results files
produced by earlier ones (e.g. `04_evaluation.ipynb` reads
`results/metrics.csv`, which `02_baselines.ipynb` and
`03_chronos2_forecasting.ipynb` populate).

**Streamlit app**:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` by default. Demo Mode works immediately
using the committed `results/*.csv` files; Live Forecast Mode additionally
requires the raw dataset to be present locally (step 4 above).

## Deployment

Deployed via **Streamlit Community Cloud**, connected directly to this
GitHub repository:

```
git add .
git commit -m "..."
git push
```

Streamlit Community Cloud automatically redeploys on every push to the
connected branch. A few deployment-specific notes:

- `requirements.txt` is pinned to the exact versions verified in local
  development (see the file itself) — Community Cloud builds a fresh
  environment from this file, so unpinned or mismatched versions can behave
  differently than local testing.
- The raw dataset (`data/raw/LD2011_2014.txt`) is **not** in this
  repository. Demo Mode does not need it. Live Forecast Mode currently
  requires it to be present in the deployed environment — if you want Live
  Mode to work on the deployed app, you'll need to either commit the dataset
  via **Git LFS** (Community Cloud supports LFS-backed repos) or fetch it at
  startup from an external source.
- `torch` is installed from the CPU wheel index, since Community Cloud
  containers have no GPU.

## Future Work

- Extend the multi-series experiment beyond 370 series' worth of _metrics_
  to full precomputed forecasts, if broader Demo Mode coverage is needed
  (currently limited to a representative subset — see `app.py` for the
  trade-off discussion)
- Add external covariates (temperature, humidity) if a matching weather
  dataset for the 2011–2014 period/region can be sourced
- Formal hyperparameter tuning for ARIMA per series, rather than a single
  fixed `(p, d, q)` used as a baseline across all experiments
- Confidence intervals / probabilistic forecasts (Chronos-2 supports
  quantile outputs) rather than point forecasts only
- Automated tests for `src/` (currently validated ad hoc through notebook
  runs)
