import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="KPI Forecasting", layout="wide")
st.title("Multivariate KPI Forecasting")

# Data sources 
DATA_DIR = Path("data")
DATASETS = {
    "Boise State": DATA_DIR / "boise_state_kpi.xlsx",
    "Eastern Oregon": DATA_DIR / "eastern_oregon_kpi.xlsx",
    "UAB": DATA_DIR / "uab_kpi.xlsx",
}

@st.cache_data
def _read_table(path_or_buf, is_xlsx: bool):
    if is_xlsx:
        return pd.read_excel(path_or_buf)
    return pd.read_csv(path_or_buf)

# Choose data 
mode = st.radio("Choose data source:", ["Use included dataset", "Upload .xlsx/.csv"], horizontal=True)
df = None

if mode == "Use included dataset":
    name = st.selectbox("Dataset", list(DATASETS.keys()))
    path = DATASETS[name]
    if not path.exists():
        st.error(f"File not found: {path}. Upload it to your repo at {path}")
        st.stop()
    df = _read_table(path, path.suffix.lower() == ".xlsx")
else:
    up = st.file_uploader("Upload file", type=["xlsx", "csv"])
    if up is not None:
        df = _read_table(up, up.name.lower().endswith(".xlsx"))

if df is None:
    st.info("Load a dataset to begin.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(), use_container_width=True)

# Column selection 
candidate_time_cols = [c for c in df.columns if c.lower() in {"year", "date"}]
time_col = st.selectbox("Time column (year or date)", candidate_time_cols or list(df.columns))

numeric_cols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.error("No numeric KPI columns found. Please upload a table with numeric KPIs.")
    st.stop()

target_col = st.selectbox("Target KPI", numeric_cols)

# Display formatting preference (counts vs. continuous)
format_as_int = st.checkbox("Format values as whole numbers (counts)", value=True)

# Clean & order 
df = df.dropna(subset=[time_col, target_col]).copy()

# coerce time to year
if np.issubdtype(df[time_col].dtype, np.datetime64):
    df["_t"] = pd.to_datetime(df[time_col]).dt.year
else:
    df["_t"] = pd.to_numeric(df[time_col], errors="coerce")
df = df.dropna(subset=["_t"]).sort_values("_t")

if df.shape[0] < 8:
    st.error("Not enough rows after cleaning. Provide more years.")
    st.stop()

st.write(f"Using **{target_col}** over {int(df['_t'].min())}–{int(df['_t'].max())}")

# Features 
lags = st.slider("Number of lags", 1, 6, 3)
horizon = st.slider("Forecast horizon (years)", 1, 10, 5)

series = df[target_col].astype(float).to_numpy()
X, y = [], []
for i in range(len(series) - lags):
    X.append(series[i:i + lags])
    y.append(series[i + lags])
X = np.asarray(X); y = np.asarray(y)

if len(X) < 10:
    st.error("Not enough data points after lagging. Reduce lags or provide more years.")
    st.stop()

# Train / eval 
split = int(0.8 * len(X))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_tr, y_tr)
pred_te = rf.predict(X_te)

mae = mean_absolute_error(y_te, pred_te)
rmse = float(mean_squared_error(y_te, pred_te) ** 0.5)  # works on any sklearn
r2 = r2_score(y_te, pred_te)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:,.0f}" if format_as_int else f"{mae:,.3f}")
c2.metric("RMSE", f"{rmse:,.0f}" if format_as_int else f"{rmse:,.3f}")
c3.metric("R²", f"{r2:.3f}")

# Forecast 
last_window = series[-lags:].tolist()
future = []
for _ in range(horizon):
    y_hat = rf.predict([last_window[-lags:]])[0]
    future.append(y_hat)
    last_window.append(y_hat)

years = df["_t"].astype(int).to_numpy()
last_year = int(years[-1])
future_years = np.arange(last_year + 1, last_year + 1 + horizon)

# Plot (bigger, cleaner) 
fig, ax = plt.subplots(figsize=(11, 5))  # bigger figure
# y-axis formatter
if format_as_int:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
else:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.2f}"))

ax.plot(years, series, label="Historical", linewidth=2)
if split < len(y):
    test_start_idx = lags + split
    if test_start_idx < len(years):
        test_years = years[test_start_idx : test_start_idx + len(pred_te)]
        ax.plot(test_years, pred_te, label="Test predictions", linewidth=1.8, linestyle="--")
ax.plot(future_years, future, "--", label="Forecast", linewidth=2)

ax.set_xlabel("Year")
ax.set_ylabel(target_col)
ax.grid(True, alpha=0.25)
ax.legend(loc="best")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

# Forecast table (rounded if desired) 
out = pd.DataFrame({"Year": future_years, f"{target_col}_forecast": future})
if format_as_int:
    out[f"{target_col}_forecast"] = np.rint(out[f"{target_col}_forecast"]).astype(int)

st.subheader("Forecast table")
st.dataframe(out, use_container_width=True)

# download button
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download forecast CSV", data=csv_bytes, file_name="forecast.csv", mime="text/csv")
