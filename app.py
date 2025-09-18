import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

st.set_page_config(page_title="KPI Forecasting", layout="wide")
st.title("Multivariate KPI Forecasting")

# Load local datasets 
DATA_DIR = Path("data")
DATASETS = {
    "Boise State": DATA_DIR / "boise_state_kpi.xlsx",
    "Eastern Oregon": DATA_DIR / "eastern_oregon_kpi.xlsx",
    "UAB": DATA_DIR / "uab_kpi.xlsx",
}

# Choose built-in dataset or upload your own
mode = st.radio("Choose data source:", ["Use included dataset", "Upload .xlsx/.csv"], horizontal=True)

df = None
if mode == "Use included dataset":
    name = st.selectbox("Dataset", list(DATASETS.keys()))
    path = DATASETS[name]
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
else:
    up = st.file_uploader("Upload file", type=["xlsx", "csv"])
    if up:
        df = pd.read_excel(up) if up.name.endswith(".xlsx") else pd.read_csv(up)

if df is None:
    st.info("Load a dataset to begin.")
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.head())

# Try to guess a time column and numeric KPI columns
candidate_time_cols = [c for c in df.columns if c.lower() in {"year","date"}]
time_col = st.selectbox("Time column (year or date)", candidate_time_cols or list(df.columns))
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
target_col = st.selectbox("Target KPI", [c for c in num_cols if c != time_col])

# Ensure time sorted
df = df.dropna(subset=[time_col, target_col]).copy()
# Coerce 'year' if it's a date
if np.issubdtype(df[time_col].dtype, np.datetime64):
    df["_t"] = pd.to_datetime(df[time_col]).dt.year
else:
    df["_t"] = pd.to_numeric(df[time_col], errors="coerce")
df = df.dropna(subset=["_t"]).sort_values("_t")

st.write(f"Using **{target_col}** as target over {df['_t'].min():.0f}–{df['_t'].max():.0f}")

# Build simple lag features for the selected KPI
lags = st.slider("Number of lags", 1, 6, 3)
horizon = st.slider("Forecast horizon (years)", 1, 10, 5)

series = df[target_col].astype(float).values
X, y = [], []
for i in range(len(series) - lags):
    X.append(series[i:i+lags])
    y.append(series[i+lags])
X = np.asarray(X); y = np.asarray(y)

if len(X) < 10:
    st.error("Not enough data points after lagging. Reduce lags or provide more years.")
    st.stop()

split = int(0.8 * len(X))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# Train RF and evaluate
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_tr, y_tr)
pred_te = rf.predict(X_te)

mae = mean_absolute_error(y_te, pred_te)
rmse = mean_squared_error(y_te, pred_te, squared=False)
r2 = r2_score(y_te, pred_te)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.3f}")
col2.metric("RMSE", f"{rmse:.3f}")
col3.metric("R²", f"{r2:.3f}")

# Recursive forecast into the future
last_window = series[-lags:].tolist()
future = []
for _ in range(horizon):
    y_hat = rf.predict([last_window[-lags:]])[0]
    future.append(y_hat)
    last_window.append(y_hat)

# Build timeline for plotting
years = df["_t"].astype(int).values
last_year = years[-1]
future_years = np.arange(last_year + 1, last_year + 1 + horizon)

fig, ax = plt.subplots()
ax.plot(years, series, label="Historical")
# test span for visual reference
if split < len(y):
    # align test preds to their years
    test_start = years[lags + split]
    test_years = np.arange(test_start, test_start + len(pred_te))
    ax.plot(test_years, pred_te, label="Test predictions")
ax.plot(future_years, future, "--", label="Forecast")
ax.set_xlabel("Year")
ax.set_ylabel(target_col)
ax.legend()
st.pyplot(fig)

st.subheader("Forecast table")
st.dataframe(pd.DataFrame({"Year": future_years, f"{target_col}_forecast": future}))
