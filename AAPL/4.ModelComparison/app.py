import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="📊 ARIMA Stock Forecast App", layout="wide")
st.title("📈 ARIMA-Based Stock Forecasting Dashboard")

# --- File Upload ---
st.sidebar.header("1️⃣ Upload Cleaned CSV File")
file = st.sidebar.file_uploader("Upload a CSV file with a 'Date' column and 'Close' prices", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)

    if 'Close' not in df.columns:
        st.error("CSV must contain a 'Close' column.")
        st.stop()

    series = df['Close']

    st.sidebar.header("2️⃣ Forecast Settings")
    forecast_steps = st.sidebar.slider("Days to Forecast", min_value=7, max_value=365, value=30)

    st.sidebar.header("3️⃣ ARIMA Parameter Selection")
    if st.sidebar.checkbox("Use Manual ARIMA Parameters"):
        p = st.sidebar.slider("p (AR)", 0, 5, 1)
        d = st.sidebar.slider("d (I)", 0, 2, 1)
        q = st.sidebar.slider("q (MA)", 0, 5, 1)
    else:
        auto_model = auto_arima(series, seasonal=False, suppress_warnings=True)
        p, d, q = auto_model.order
        st.sidebar.success(f"Auto ARIMA Parameters: (p={p}, d={d}, q={q})")

    # --- Accuracy Evaluation ---
    train = series[:-forecast_steps]
    test = series[-forecast_steps:]

    model_eval = ARIMA(train, order=(p, d, q)).fit()
    # Align eval forecasts to test dates
    pred_eval = pd.Series(
        model_eval.forecast(steps=forecast_steps).values,
        index=test.index,
        name="Forecast"
    )

    rmse = np.sqrt(mean_squared_error(test, pred_eval))
    mae = mean_absolute_error(test, pred_eval)
    mape = np.mean(np.abs((test - pred_eval) / test)) * 100
    r2 = r2_score(test, pred_eval)

    # --- Forecast Future ---
    model = ARIMA(series, order=(p, d, q)).fit()
    forecast = pd.Series(
        model.forecast(steps=forecast_steps).values,
        index=pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B'),
        name="Forecast"
    )
    forecast_df = pd.DataFrame({"Forecast": forecast})

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Raw Data", "🔮 Forecast Plot", "📆 Trend & Seasonality",
        "📊 Actual vs Forecast", "📄 Forecast Table"
    ])

    with tab1:
        st.subheader("📁 Raw Data")
        st.line_chart(series)

    with tab2:
        st.subheader(f"🔮 Forecast Plot ({forecast_steps} Days)")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series.index, series, label="Historical", color='blue')
        ax.plot(forecast.index, forecast, label="Forecast", linestyle="--", color='red')
        ax.fill_between(forecast.index, forecast * 0.95, forecast * 1.05, alpha=0.2, color='orange')
        ax.set_title(f"ARIMA({p},{d},{q}) Forecast")
        ax.legend()
        st.pyplot(fig)

    with tab3:
        st.subheader("📆 Weekly & Yearly Seasonality")
        df['Weekday'] = df.index.day_name()
        df['Month'] = df.index.month
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        df.groupby('Weekday')['Close'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
        ]).plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title("Weekly Trend")
        df.groupby('Month')['Close'].mean().plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title("Monthly Trend")
        st.pyplot(fig)

    with tab4:
        st.subheader("📊 Actual vs Forecast")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test.index, test, label="Actual", color='green')
        ax.plot(pred_eval.index, pred_eval, label="Predicted", linestyle="--", color='red')
        ax.fill_between(pred_eval.index, pred_eval * 0.95, pred_eval * 1.05, alpha=0.2, color='gray')
        ax.set_title("Validation Forecast")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📈 Forecast Accuracy")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RMSE", f"{rmse:.2f}")
        c2.metric("MAE", f"{mae:.2f}")
        c3.metric("MAPE", f"{mape:.2f}%")
        c4.metric("R² Score", f"{r2:.2f}")
        c5.subheader("🎯 Accuracy")
        c5.markdown(f"**Horizon:** {forecast_steps} days")
        c5.markdown(f"**Train size:** {len(train)}")
        c5.markdown(f"**Test size:** {len(test)}")

        st.markdown("### 📌 Individual Accuracy Breakdown")
        acc_tbl = pd.DataFrame({
            "Actual": test,
            "Pred": pred_eval
        }, index=test.index)
        acc_tbl["Error"] = acc_tbl["Actual"] - acc_tbl["Pred"]
        acc_tbl["APE%"] = np.abs(acc_tbl["Error"] / acc_tbl["Actual"]) * 100
        st.dataframe(acc_tbl.round(2))

    with tab5:
        st.subheader("📄 Forecast Table")
        st.dataframe(forecast_df)
        st.download_button("Download Forecast CSV", data=forecast_df.to_csv().encode(), file_name="arima_forecast.csv")

else:
    st.info("👈 Please upload a CSV file to begin.")
