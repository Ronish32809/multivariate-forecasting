# STEP 1: IMPORTS & GLOBAL DEFAULTS

from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# default forecast knobs
DEFAULT_FORECAST_YEARS = 5
DEFAULT_RF_LAGS       = 4
DEFAULT_RF_TREES      = 200
SPLIT_RATIO           = 0.80
MIN_TEST_POINTS       = 2

# LSTM setup — learning rate is fixed on purpose
LEARNING_RATE   = 0.001
DEFAULT_LSTM_UNITS = 32
DEFAULT_LSTM_ACT   = "tanh"


# PATHS & COLUMN NORMALIZATION (maps R&D header variations)
DATA_DIR = Path("data")
EXCEL_PATHS = {
    "bsu":             DATA_DIR / "boise_state_kpi.xlsx",
    "eastern_oregon":  DATA_DIR / "eastern_oregon_kpi.xlsx",
    "uab":             DATA_DIR / "uab_kpi.xlsx",
}

# normalize headers so the rest of the code is simple
COLUMN_RENAME = {
    "R&D Expenditure (Millions USD)": "R&D_MUSD",
    "R&D expenditure (Millions USD)": "R&D_MUSD",
    "R&D Expenditure": "R&D_MUSD",
}

# labels used in plots/tables
LABELS = {
    "R&D_MUSD":        "R&D Expenditure",
    "Enrollment":      "Enrollment",
    "Employees":       "Employees",
    "Total_Graduates": "Total Graduates",
}


# OPTIONAL TENSORFLOW (if not installed, we just skip LSTM and keep RF)
use_lstm = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K
    tf.config.run_functions_eagerly(True)
except Exception:
    use_lstm = False  # mirrors the notebook behavior

np.random.seed(42)


# STREAMLIT SIDEBAR 
st.set_page_config(page_title="University KPI Forecasts", layout="wide")
st.title("University KPI Forecasts")

st.sidebar.header("Controls")
university_choice = st.sidebar.selectbox("University", ["bsu", "eastern_oregon", "uab", "all"], index=0)
kpi_to_run = st.sidebar.selectbox(
    "KPI",
    ["Enrollment", "Employees", "R&D Expenditure", "Total Graduates"],
    index=0,
)

with st.sidebar.expander("Parameters", expanded=True):
    forecast_years = st.number_input("Forecast years", 1, 20, DEFAULT_FORECAST_YEARS, 1)
    rf_lags        = st.number_input("RF lags (lookback)", 1, 24, DEFAULT_RF_LAGS, 1)
    rf_estimators  = st.number_input("RF trees (n_estimators)", 10, 1000, DEFAULT_RF_TREES, 10)

    # LSTM hyperparams (user can play with these; LR remains fixed)
    lstm_units      = st.number_input("LSTM units", 8, 256, DEFAULT_LSTM_UNITS, 8)
    lstm_activation = st.selectbox("LSTM activation", ["tanh", "relu", "sigmoid"], index=0)

    st.caption(f"Train/Test split: {int(SPLIT_RATIO*100)}/{int((1-SPLIT_RATIO)*100)} · Min test pts: {MIN_TEST_POINTS}")
    st.caption(f"LSTM learning rate (fixed): {LEARNING_RATE}")

run = st.sidebar.button("Run forecast")


# STEP 2: HELPERS S

def make_optimizer(lr=LEARNING_RATE):
    return Adam(learning_rate=lr)

def mape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0

def metric_row(name, y_true, y_pred):
    return {
        "Model": name,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE%": mape(y_true, y_pred),
    }

def split_index(n, split=SPLIT_RATIO, min_test=MIN_TEST_POINTS, min_train=None):
    # guarantee we have at least a couple of train rows and at least some test rows
    if min_train is None:
        min_train = rf_lags + 3
    si = max(int(n * split), min_train)
    if n - si < min_test:
        si = n - min_test
    si = max(1, min(si, n - 1))
    return si

def y_axis_label(label_text):
    if "R&D" in label_text:       return "R&D Spending (Millions USD)"
    if "Graduate" in label_text:  return "Graduates (count)"
    if "Enrollment" in label_text:return "Students (count)"
    if "Employee" in label_text:  return "Employees (count)"
    return label_text

def round_like_humans(x, label_text):
    # counts → integers; money → two decimals
    if any(k in label_text for k in ["Enrollment","Employee","Graduate","Students"]):
        return int(np.round(x))
    return float(np.round(x, 2))

def load_excel_kpis(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Missing data file: {excel_path} (CWD={os.getcwd()})")

    df = pd.read_excel(excel_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=COLUMN_RENAME)

    keep = ["Date", "Enrollment", "Employees", "R&D_MUSD", "Total_Graduates"]
    keep = [c for c in keep if c in df.columns]
    if "Date" not in keep:
        raise KeyError("No 'Date' column found in the workbook.")

    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    # make numeric + fill tiny gaps
    for col in [c for c in df.columns if c != "Date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").interpolate("linear").ffill().bfill()
    return df

def choose_series(df_one_uni: pd.DataFrame, kpi_name: str):
    if kpi_name == "R&D Expenditure":
        if "R&D_MUSD" in df_one_uni.columns:
            return df_one_uni["R&D_MUSD"], LABELS["R&D_MUSD"]
        raise KeyError("'R&D Expenditure' not found in this file.")
    if kpi_name == "Total Graduates":
        if "Total_Graduates" in df_one_uni.columns:
            return df_one_uni["Total_Graduates"], LABELS["Total_Graduates"]
        raise KeyError("'Total Graduates' not found in this file.")
    if kpi_name == "Enrollment":
        if "Enrollment" in df_one_uni.columns:
            return df_one_uni["Enrollment"], LABELS["Enrollment"]
        raise KeyError("'Enrollment' not found in this file.")
    if kpi_name == "Employees":
        if "Employees" in df_one_uni.columns:
            return df_one_uni["Employees"], LABELS["Employees"]
        raise KeyError("'Employees' not found in this file.")
    raise KeyError(f"Unknown KPI: {kpi_name}")


# STEP 3: MULTIVARIATE VALIDATION + MULTIVARIATE FORECASTS

def validate_series_multivariate(series: pd.Series, pretty_label: str, df_all_cols: pd.DataFrame,
                                 lstm_units_: int, lstm_activation_: str):
    """80/20 validation:
       • RF: lags of target + contemporaneous covariates
       • LSTM: window = [target + other KPIs] → next target
       • Naive: last value
    """
    s = series.sort_index().astype(float).interpolate().ffill().bfill()
    n = len(s)
    if n < max(rf_lags + 3, 8):
        return None

    si = split_index(n)
    train, test = s.iloc[:si], s.iloc[si:]
    y_true = test.values
    rows = []

    # Naive baseline
    rows.append(metric_row("Naive(last)", y_true, np.repeat(train.iloc[-1], len(test))))

    # RF (multivariate)
    covars = df_all_cols.drop(columns=[series.name])
    lag = pd.DataFrame({"y": s})
    for i in range(1, rf_lags + 1):
        lag[f"lag_{i}"] = lag["y"].shift(i)
    X_full = lag.join(covars, how="inner").dropna()

    if not X_full.empty:
        X_tr = X_full.iloc[:si].drop(columns=["y"]).values
        y_tr = X_full.iloc[:si]["y"].values
        rf_model = RandomForestRegressor(n_estimators=rf_estimators, random_state=42)
        rf_model.fit(X_tr, y_tr)

        preds, hist_target = [], s.iloc[:si].copy()
        idx_test = s.index[si:]
        for t in range(len(idx_test)):
            lag_vals = hist_target.iloc[-rf_lags:].values
            cov_vals = covars.loc[idx_test[t]].values
            x = np.concatenate([lag_vals, cov_vals]).reshape(1, -1)
            y_hat = rf_model.predict(x)[0]
            preds.append(round_like_humans(y_hat, pretty_label))
            hist_target = pd.concat([hist_target, pd.Series([preds[-1]], index=[idx_test[t]])])
        rows.append(metric_row("Random Forest", y_true, np.array(preds)))

    # LSTM (multivariate)
    if use_lstm:
        target_col = series.name
        covar_cols = [c for c in df_all_cols.columns if c != target_col]

        scaler_y = MinMaxScaler().fit(train.values.reshape(-1, 1))
        scaler_x = MinMaxScaler().fit(df_all_cols.loc[train.index, covar_cols].values)

        y_tr_s = scaler_y.transform(train.values.reshape(-1, 1))
        X_tr_s = scaler_x.transform(df_all_cols.loc[train.index, covar_cols].values)
        feats  = np.concatenate([y_tr_s, X_tr_s], axis=1)

        X_l, y_l = [], []
        for i in range(len(feats) - rf_lags):
            X_l.append(feats[i:i + rf_lags, :])
            y_l.append(y_tr_s[i + rf_lags, 0])
        if len(X_l) > 0:
            X_l = np.array(X_l); y_l = np.array(y_l).reshape(-1, 1)

            K.clear_session()
            lstm = Sequential([
                LSTM(lstm_units_, activation=lstm_activation_, input_shape=(rf_lags, X_l.shape[2])),
                Dropout(0.1),
                Dense(1)
            ])
            lstm.compile(optimizer=make_optimizer(), loss="mse")
            _ = lstm.fit(
                X_l, y_l, epochs=200, validation_split=0.25, verbose=0,
                callbacks=[EarlyStopping(patience=12, restore_best_weights=True)]
            )

            X_te_s = scaler_x.transform(df_all_cols.loc[test.index, covar_cols].values)
            window = np.concatenate([y_tr_s, X_tr_s], axis=1)[-rf_lags:, :].copy()

            lstm_preds = []
            for k in range(len(test)):
                nxt_s = float(lstm.predict(window.reshape(1, rf_lags, -1), verbose=0).squeeze())
                next_row = np.concatenate([[nxt_s], X_te_s[k]]).astype(float)
                window   = np.vstack([window[1:], next_row])
                y_inv = scaler_y.inverse_transform(np.array([[nxt_s]])).ravel()[0]
                lstm_preds.append(round_like_humans(y_inv, pretty_label))
            rows.append(metric_row("LSTM", y_true, np.array(lstm_preds)))

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return metrics


def predict_series_multivariate(series: pd.Series, pretty_label: str, df_all_cols: pd.DataFrame,
                                years_ahead: int, lstm_units_: int, lstm_activation_: str):
    """returns (rf_forecast, lstm_forecast_or_None, future_index)"""
    s = series.sort_index().astype(float).interpolate().ffill().bfill()
    future_index = pd.date_range(start=f"{s.index.max().year+1}-01-01",
                                 periods=years_ahead, freq="YS")

    # RF (multivariate)
    other_cols = df_all_cols.drop(columns=[series.name])
    lag = pd.DataFrame({"y": s})
    for i in range(1, rf_lags + 1):
        lag[f"lag_{i}"] = lag["y"].shift(i)
    X_full = lag.join(other_cols, how="inner").dropna()

    X = X_full.drop(columns=["y"]).values
    y = X_full["y"].values
    rf_model = RandomForestRegressor(n_estimators=rf_estimators, random_state=42)
    rf_model.fit(X, y)

    history_for_rf = s.copy()
    last_cov = other_cols.iloc[-1].values  # last known covariates carried forward
    rf_fore = []
    for _ in range(years_ahead):
        last_lags = history_for_rf.iloc[-rf_lags:].values
        x_in = np.concatenate([last_lags, last_cov]).reshape(1, -1)
        y_hat = rf_model.predict(x_in)[0]
        rf_fore.append(round_like_humans(y_hat, pretty_label))
        history_for_rf.loc[history_for_rf.index.max() + pd.offsets.YearBegin()] = rf_fore[-1]
    rf_fore = np.array(rf_fore)

    # LSTM (multivariate)
    lstm_fore = None
    if use_lstm:
        target_col = series.name
        covar_cols = [c for c in df_all_cols.columns if c != target_col]

        sc_y = MinMaxScaler().fit(s.values.reshape(-1, 1))
        sc_x = MinMaxScaler().fit(df_all_cols.loc[s.index, covar_cols].values)

        y_all_s = sc_y.transform(s.values.reshape(-1, 1))
        X_all_s = sc_x.transform(df_all_cols.loc[s.index, covar_cols].values)
        feats = np.concatenate([y_all_s, X_all_s], axis=1)

        X_l, y_l = [], []
        for i in range(len(feats) - rf_lags):
            X_l.append(feats[i:i + rf_lags, :])
            y_l.append(y_all_s[i + rf_lags, 0])

        if len(X_l) > 0:
            X_l = np.array(X_l); y_l = np.array(y_l).reshape(-1, 1)

            K.clear_session()
            lstm = Sequential([
                LSTM(lstm_units_, activation=lstm_activation_, input_shape=(rf_lags, X_l.shape[2])),
                Dropout(0.1),
                Dense(1)
            ])
            lstm.compile(optimizer=make_optimizer(LEARNING_RATE), loss="mse")
            _ = lstm.fit(
                X_l, y_l, epochs=200, validation_split=0.25, verbose=0,
                callbacks=[EarlyStopping(patience=12, restore_best_weights=True)]
            )

            last_cov_s = sc_x.transform(df_all_cols.loc[[s.index[-1]], covar_cols].values)[0]
            window = feats[-rf_lags:, :].copy()
            preds_s = []
            for _ in range(years_ahead):
                nxt_s = float(lstm.predict(window.reshape(1, rf_lags, -1), verbose=0).squeeze())
                next_row = np.concatenate([[nxt_s], last_cov_s]).astype(float)
                window = np.vstack([window[1:], next_row])
                preds_s.append(nxt_s)

            lstm_raw = sc_y.inverse_transform(np.array(preds_s).reshape(-1, 1)).ravel()
            lstm_fore = np.array([round_like_humans(v, pretty_label) for v in lstm_raw])

    return rf_fore, lstm_fore, future_index


# PLOTTING

def plot_single(series, label, rf_fore, lstm_fore, future_idx, uni_key):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, 'o-', label="Actual (history)")
    ax.plot([series.index[-1]] + list(future_idx), [series.values[-1]] + list(rf_fore), '--x', label="Random Forest")
    if lstm_fore is not None:
        ax.plot([series.index[-1]] + list(future_idx), [series.values[-1]] + list(lstm_fore), ':o', label="LSTM")
    ax.set_title(f"{uni_key.upper()} — {label} ({future_idx.year[0]}–{future_idx.year[-1]})")
    ax.set_xlabel("Year"); ax.set_ylabel(y_axis_label(label))
    ax.grid(True); ax.legend(ncol=3); fig.tight_layout()
    return fig

def plot_overlay(all_lines, kpi_name, y_label, year0, year1):
    fig, ax = plt.subplots(figsize=(12, 5))
    for item in all_lines:
        uni = item["uni"]; series = item["series"]
        rf_fore = item["rf"]; lstm_fore = item["lstm"]; future_idx = item["future"]
        ax.plot(series.index, series.values, 'o-', label=f"{uni.upper()} Actual")
        ax.plot([series.index[-1]] + list(future_idx), [series.values[-1]] + list(rf_fore), '--x', label=f"{uni.upper()} RF")
        if lstm_fore is not None:
            ax.plot([series.index[-1]] + list(future_idx), [series.values[-1]] + list(lstm_fore), ':o', label=f"{uni.upper()} LSTM")
    ax.set_title(f"{kpi_name} — All Universities ({year0}–{year1})")
    ax.set_xlabel("Year"); ax.set_ylabel(y_axis_label(y_label))
    ax.grid(True); ax.legend(ncol=3); fig.tight_layout()
    return fig


# STEP 4: DISPATCH / RENDER

if not run:
    st.info("Choose options and click **Run forecast**.")
    st.stop()

try:
    # single university
    if university_choice != "all":
        df_uni = load_excel_kpis(EXCEL_PATHS[university_choice])
        series, label = choose_series(df_uni, kpi_to_run)

        metrics = validate_series_multivariate(series, label, df_uni, lstm_units, lstm_activation)
        st.subheader(f"80/20 Validation — {university_choice.upper()} — {label}")
        if metrics is not None and not metrics.empty:
            st.dataframe(metrics.round(4), use_container_width=True)
        else:
            st.caption("Not enough data to run a robust validation (still forecasting below).")

        rf_fore, lstm_fore, future_idx = predict_series_multivariate(
            series, label, df_uni, forecast_years, lstm_units, lstm_activation
        )
        st.pyplot(plot_single(series, label, rf_fore, lstm_fore, future_idx, university_choice), clear_figure=True)

        out = pd.DataFrame({"Random Forest": rf_fore}, index=future_idx)
        if lstm_fore is not None:
            out["LSTM"] = lstm_fore
        st.dataframe(out.round(2), use_container_width=True)

    # all universities
    else:
        schools = ["bsu", "eastern_oregon", "uab"]
        lines, tables, val_rows, any_future = [], [], [], None
        last_label = kpi_to_run

        for uni_key in schools:
            try:
                df_uni = load_excel_kpis(EXCEL_PATHS[uni_key])
                series, label = choose_series(df_uni, kpi_to_run)
                last_label = label

                metrics = validate_series_multivariate(series, label, df_uni, lstm_units, lstm_activation)
                if metrics is not None and not metrics.empty:
                    tmp = metrics.copy()
                    tmp.insert(0, "University", uni_key.upper())
                    tmp.insert(1, "KPI", label)
                    val_rows.append(tmp)

                rf_fore, lstm_fore, future_idx = predict_series_multivariate(
                    series, label, df_uni, forecast_years, lstm_units, lstm_activation
                )
                any_future = future_idx  # hold any valid index for title

                lines.append({"uni": uni_key, "series": series, "rf": rf_fore, "lstm": lstm_fore, "future": future_idx})

                t = pd.DataFrame({"Random Forest": rf_fore}, index=future_idx)
                if lstm_fore is not None:
                    t["LSTM"] = lstm_fore
                tables.append(t.add_prefix(f"{uni_key.upper()} "))

            except Exception as e:
                # concise per-school message (no attribute typos)
                st.warning(f"{uni_key.upper()}: {str(e)}")
                continue

        # guard against the DatetimeIndex truthiness error
        if any_future is not None and len(any_future) > 0 and len(lines) > 0:
            st.pyplot(
                plot_overlay(lines, kpi_to_run, last_label, any_future.year[0], any_future.year[-1]),
                clear_figure=True
            )

        if tables:
            st.subheader("Combined Forecast Table")
            st.dataframe(pd.concat(tables, axis=1).round(2), use_container_width=True)

        if val_rows:
            big_val = pd.concat(val_rows, axis=0, ignore_index=True)
            big_val = big_val.sort_values(["University", "KPI", "RMSE"]).reset_index(drop=True)
            st.subheader("80/20 Validation — All Universities")
            st.dataframe(big_val.round(4), use_container_width=True)

except Exception as e:
    st.error(str(e))
    st.stop()

st.caption("Tip: Keep headers exactly: Date, Enrollment, Employees, Total_Graduates, and 'R&D Expenditure (Millions USD)'. R&D is auto-mapped to R&D_MUSD.")
