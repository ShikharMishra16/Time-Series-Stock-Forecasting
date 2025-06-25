import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="📊 Model Comparison Dashboard", layout="wide")
st.title("🔍 Forecasting Model Comparison: ARIMA vs Prophet vs SARIMA")

# --- Sidebar: Upload & Settings ---
st.sidebar.header("📁 Upload & Forecast Settings")
file = st.sidebar.file_uploader("Upload CSV with 'Date' and 'Close' columns", type=["csv"])
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 180, 60)

# --- Main Logic ---
# --- Main Logic ---
if file:
    df = pd.read_csv(file)
    # Handle if 'Date' column missing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
    else:
        # Assume first column is date
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.dropna(subset=[date_col], inplace=True)
        df.set_index(date_col, inplace=True)
        df.index.name = 'Date'

    if 'Close' not in df.columns:
        st.error("CSV must contain a 'Close' column.")
        st.stop()

    close = df['Close']
    train = close[:-forecast_days]
    test = close[-forecast_days:]
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)

    if 'Close' not in df.columns:
        st.error("CSV must contain a 'Close' column.")
        st.stop()

    close = df['Close']
    train = close[:-forecast_days]
    test = close[-forecast_days:]

    # ARIMA
    try:
        arima_model = ARIMA(train, order=(1, 1, 1)).fit()
        arima_forecast = pd.Series(
            arima_model.forecast(steps=forecast_days).values,
            index=test.index,
            name='ARIMA_Forecast'
        )
    except Exception:
        arima_forecast = pd.Series([np.nan] * forecast_days, index=test.index)

    # Prophet
    df_prophet = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    prophet = Prophet()
    prophet.fit(df_prophet[:-forecast_days])
    future = prophet.make_future_dataframe(periods=forecast_days)
    forecast_prophet_full = prophet.predict(future)
    # align prophet forecast to test dates
    prophet_forecast = pd.Series(
        forecast_prophet_full['yhat'].iloc[-forecast_days:].values,
        index=test.index,
        name='Prophet_Forecast'
    )

    # SARIMA
    try:
        sarima_model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        sarima_forecast = pd.Series(
            sarima_model.forecast(steps=forecast_days).values,
            index=test.index,
            name='SARIMA_Forecast'
        )
    except Exception:
        sarima_forecast = pd.Series([np.nan] * forecast_days, index=test.index)

    # Evaluation
    def get_metrics(true, pred):
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        mape = np.mean(np.abs((true - pred) / true)) * 100
        return rmse, mae, r2, mape

    rmse_a, mae_a, r2_a, mape_a = get_metrics(test, arima_forecast)
    rmse_p, mae_p, r2_p, mape_p = get_metrics(test, prophet_forecast)
    rmse_s, mae_s, r2_s, mape_s = get_metrics(test, sarima_forecast)

    # Metrics Table
    st.subheader("📊 Evaluation Metrics")
    metrics_df = pd.DataFrame({
        'Model': ['ARIMA', 'Prophet', 'SARIMA'],
        'RMSE': [rmse_a, rmse_p, rmse_s],
        'MAE': [mae_a, mae_p, mae_s],
        'R²': [r2_a, r2_p, r2_s],
        'MAPE (%)': [mape_a, mape_p, mape_s]
    })
    st.dataframe(metrics_df.round(2))

        # Enhanced Forecast Plot
    st.subheader("🎨 Enhanced Forecast Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot full historical series smoothly
    ax.plot(close.index, close, label='Historical', color='gray', linewidth=1)
    # Plot actual test values
    ax.plot(test.index, test, label='Actual (Test)', color='black', linewidth=2)
    # Plot forecasts only for test period
    ax.plot(test.index, arima_forecast, label='ARIMA Forecast', linestyle='--', color='orange')
    ax.plot(test.index, prophet_forecast, label='Prophet Forecast', linestyle=':', color='green')
    ax.plot(test.index, sarima_forecast, label='SARIMA Forecast', linestyle='-.', color='blue')
    ax.axvline(test.index[0], color='red', linestyle=':', label='Forecast Start')
    ax.set_title('Forecast Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
