import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(layout="wide", page_title="📊 Multi-Model Stock Forecast")
st.title("📈 Forecast Comparison: Prophet vs ARIMA vs SARIMA")

# Upload CSV
data = st.sidebar.file_uploader("Upload a CSV file with 'Date' and 'Close' columns", type=["csv"])
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 180, 60)

if data:
    df = pd.read_csv(data)[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(subset=['ds', 'y'], inplace=True)

    st.subheader("📈 Historical Data")
    st.line_chart(df.set_index('ds')['y'])

    # ===================== Prophet =====================
    prophet_df = df.copy()
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=forecast_days)
    prophet_forecast = prophet_model.predict(future)
    prophet_merge = pd.merge(prophet_df, prophet_forecast[['ds', 'yhat']], on='ds', how='left')

    prophet_mae = mean_absolute_error(prophet_merge['y'], prophet_merge['yhat'])
    prophet_rmse = mean_squared_error(prophet_merge['y'], prophet_merge['yhat']) ** 0.5
    prophet_mape = np.mean(np.abs((prophet_merge['y'] - prophet_merge['yhat']) / prophet_merge['y'])) * 100
    prophet_acc = 100 - prophet_mape

    # ===================== ARIMA =====================
    arima_series = df.set_index('ds')['y']
    arima_model = ARIMA(arima_series, order=(5,1,0)).fit()
    arima_pred = arima_model.predict(start=len(arima_series)-forecast_days, end=len(arima_series)-1, typ='levels')
    arima_actual = arima_series[-forecast_days:]

    arima_mae = mean_absolute_error(arima_actual, arima_pred)
    arima_rmse = mean_squared_error(arima_actual, arima_pred)**0.5
    arima_mape = np.mean(np.abs((arima_actual - arima_pred) / arima_actual)) * 100
    arima_acc = 100 - arima_mape

    # ===================== SARIMA =====================
    sarima_model = SARIMAX(arima_series, order=(1,1,1), seasonal_order=(1,0,1,12)).fit(disp=False)
    sarima_pred = sarima_model.forecast(steps=forecast_days)
    sarima_actual = arima_series[-forecast_days:]

    sarima_mae = mean_absolute_error(sarima_actual, sarima_pred)
    sarima_rmse = mean_squared_error(sarima_actual, sarima_pred)**0.5
    sarima_mape = np.mean(np.abs((sarima_actual - sarima_pred) / sarima_actual)) * 100
    sarima_acc = 100 - sarima_mape

    # ===================== Plot Comparison =====================
    st.subheader("🔍 Forecast Comparison (Last N Days)")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['ds'][-forecast_days:], df['y'][-forecast_days:], label='Actual', color='black')
    ax.plot(df['ds'][-forecast_days:], prophet_merge['yhat'][-forecast_days:], label='Prophet Forecast', linestyle='--')
    ax.plot(df['ds'][-forecast_days:], arima_pred, label='ARIMA Forecast', linestyle='--')
    ax.plot(df['ds'][-forecast_days:], sarima_pred, label='SARIMA Forecast', linestyle='--')
    ax.set_title("Forecast Comparison")
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

    # ===================== Metrics Table =====================
    st.subheader("📊 Evaluation Metrics")
    metrics_df = pd.DataFrame({
        'Model': ['Prophet', 'ARIMA', 'SARIMA'],
        'MAE': [prophet_mae, arima_mae, sarima_mae],
        'RMSE': [prophet_rmse, arima_rmse, sarima_rmse],
        'MAPE (%)': [prophet_mape, arima_mape, sarima_mape],
        'Accuracy (%)': [prophet_acc, arima_acc, sarima_acc]
    }).round(2)
    st.dataframe(metrics_df.set_index('Model'))

else:
    st.warning("Please upload a valid CSV file with 'Date' and 'Close' columns.")
