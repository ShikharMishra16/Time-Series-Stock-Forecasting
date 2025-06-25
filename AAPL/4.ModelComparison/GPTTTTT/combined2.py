import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="📈 Time Series Forecasting", layout="wide")
st.title("📊 Forecasting Dashboard (ARIMA • SARIMA • Prophet)")

st.sidebar.header("📂 Upload Time Series CSV")
uploaded = st.sidebar.file_uploader("Upload CSV with 'Date' and 'Close' columns", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)

    if 'Close' not in df.columns:
        st.error("CSV must have a 'Close' column.")
        st.stop()

    st.subheader("📁 Uploaded Data")
    st.line_chart(df['Close'])

    forecast_days = st.slider("Forecast Horizon (days)", 30, 180, 60, step=10)

    tab1, tab2, tab3 = st.tabs(["ARIMA", "SARIMA", "Prophet"])

    with tab1:
        st.subheader("🔮 ARIMA Forecast")
        series = df['Close']
        train = series[:-forecast_days]
        test = series[-forecast_days:]

        arima_model = auto_arima(series, seasonal=False, suppress_warnings=True)
        p, d, q = arima_model.order

        model = ARIMA(train, order=(p, d, q)).fit()
        forecast_arima = model.forecast(steps=forecast_days)
        forecast_arima.index = test.index

        st.line_chart(pd.concat([test, forecast_arima], axis=1).rename(columns={"Close": "Actual", 0: "Forecast"}))

        mae = mean_absolute_error(test, forecast_arima)
        rmse = mean_squared_error(test, forecast_arima) ** 0.5
        mape = np.mean(np.abs((test - forecast_arima) / test)) * 100
        r2 = r2_score(test, forecast_arima)
        accuracy = 100 - mape

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("R²", f"{r2:.4f}")
        col4.metric("MAPE", f"{mape:.2f}%")
        col5.metric("Accuracy", f"{accuracy:.2f}%")

    with tab2:
        st.subheader("📈 SARIMA Forecast")
        log_series = np.log(df['Close'])
        d_order = 0
        while d_order < 3:
            p_val = adfuller(log_series.dropna())[1]
            if p_val < 0.05:
                break
            log_series = log_series.diff().dropna()
            d_order += 1

        sarima_train = log_series[:-forecast_days]
        sarima_test = log_series[-forecast_days:]
        p, d, q = 1, d_order, 1
        P, D, Q, s = 0, 1, 1, 12

        sarima_model = SARIMAX(sarima_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
        sarima_fit = sarima_model.fit(disp=False)
        sarima_pred = sarima_fit.forecast(steps=forecast_days)
        sarima_pred.index = sarima_test.index

        log_base = df['Close'].iloc[-forecast_days - 1]
        inv_pred = np.exp(np.log(log_base) + sarima_pred.cumsum())
        inv_actual = df['Close'].iloc[-forecast_days:]

        st.line_chart(pd.concat([inv_actual, inv_pred], axis=1).rename(columns={"Close": "Actual", 0: "Forecast"}))

        mae = mean_absolute_error(inv_actual, inv_pred)
        rmse = mean_squared_error(inv_actual, inv_pred) ** 0.5
        mape = np.mean(np.abs((inv_actual - inv_pred) / inv_actual)) * 100
        r2 = r2_score(inv_actual, inv_pred)
        accuracy = 100 - mape

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("R²", f"{r2:.4f}")
        col4.metric("MAPE", f"{mape:.2f}%")
        col5.metric("Accuracy", f"{accuracy:.2f}%")

    with tab3:
        st.subheader("📅 Prophet Forecast")
        prophet_df = df.reset_index()[['Date', 'Close']]
        prophet_df.columns = ['ds', 'y']
        m = Prophet()
        m.fit(prophet_df)

        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)

        fig1 = m.plot(forecast)
        st.pyplot(fig1)

        merged = prophet_df.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        eval_window = merged.dropna().tail(forecast_days)

        mae = mean_absolute_error(eval_window['y'], eval_window['yhat'])
        rmse = mean_squared_error(eval_window['y'], eval_window['yhat']) ** 0.5
        r2 = r2_score(eval_window['y'], eval_window['yhat'])
        mape = (np.abs((eval_window['y'] - eval_window['yhat']) / eval_window['y'])).mean() * 100
        accuracy = 100 - mape

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("R²", f"{r2:.4f}")
        col4.metric("MAPE", f"{mape:.2f}%")
        col5.metric("Accuracy", f"{accuracy:.2f}%")

else:
    st.info("👈 Please upload a CSV to begin.")

    st.info("👈 Upload a dataset or fetch from ticker to begin.")
