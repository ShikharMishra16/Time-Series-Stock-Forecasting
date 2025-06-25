import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# Preprocessing Function
def preprocess_data(ticker="MSFT", period="5y", apply_log=True, max_diff=3):
    df = yf.download(ticker, period=period)
    if df.empty:
        st.error("❌ Could not fetch stock data. Check ticker or network.")
        st.stop()

    df = df[['Close']].dropna()
    df.columns = ['Close']
    df.index = pd.to_datetime(df.index)

    df['LogClose'] = np.log(df['Close']) if apply_log else df['Close']
    series = df['LogClose'].copy()
    differencing_order = 0

    def is_stationary(s):
        return adfuller(s.dropna())[1] < 0.05

    while not is_stationary(series) and differencing_order < max_diff:
        series = series.diff().dropna()
        differencing_order += 1

    return df, series, differencing_order

# Split Data
def train_test_split(series, test_size=40):
    return series[:-test_size], series[-test_size:]

# Train SARIMA
def train_sarima(train, order, seasonal_order):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

# Evaluation
def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    st.subheader("📊 Evaluation Metrics")
    st.markdown(f"- **MAE**: {mae:.4f}")
    st.markdown(f"- **RMSE**: {rmse:.4f}")
    st.markdown(f"- **R²**: {r2:.4f}")
    return mae, rmse, r2

# Download CSV
def download_csv(df, filename="forecast.csv"):
    buffer = BytesIO()
    df.to_csv(buffer, index=True)
    buffer.seek(0)
    st.download_button("⬇️ Download Forecast CSV", buffer, file_name=filename, mime="text/csv")

# Streamlit App
def main():
    st.title("📊 SARIMA-Based Stock Forecasting")

    ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    forecast_days = st.slider("Forecast Days", 30, 180, step=10, value=60)
    period = st.selectbox("Select Data Period", ["1y", "2y", "5y"], index=0)
    apply_log = st.checkbox("Apply Log Transformation", value=True)

    if st.button("Start Forecasting"):
        df, series, d_order = preprocess_data(ticker, period, apply_log)
        train, test = train_test_split(series, forecast_days)

        st.line_chart(df['Close'])

        st.subheader("🔧 SARIMA Parameters")
        col1, col2 = st.columns(2)
        with col1:
            p = st.number_input("p", 0, 5, 1)
            d = d_order
            q = st.number_input("q", 0, 5, 1)
        with col2:
            P = st.number_input("P", 0, 2, 0)
            D = st.number_input("D", 0, 2, 0)
            Q = st.number_input("Q", 0, 2, 0)
            s = st.number_input("Seasonal Period (s)", 1, 365, 12)

        with st.spinner("Training SARIMA Model..."):
            model = train_sarima(train, (p, d, q), (P, D, Q, s))

            # Aligned forecast series
            sarima_forecast = pd.Series(
                model.forecast(steps=forecast_days).values,
                index=test.index,
                name="SARIMA_Forecast"
            )

            # Confidence intervals
            conf_int = model.get_forecast(steps=forecast_days).conf_int()
            conf_int.index = test.index

            # Evaluate
            evaluate_model(test, sarima_forecast)

            # Inverse transform if needed
            if apply_log:
                last_log = df['LogClose'].iloc[-forecast_days - 1]
                pred_log = last_log + sarima_forecast.cumsum()
                forecast_vals = np.exp(pred_log)
                actual_vals = np.exp(df['LogClose'][-forecast_days:])
            else:
                forecast_vals = sarima_forecast
                actual_vals = df['Close'][-forecast_days:]

            forecast_df = pd.DataFrame({
                "Forecasted Price": forecast_vals,
                "Actual Price": actual_vals
            }, index=test.index)

            st.subheader("📈 Forecast Plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(actual_vals, label="Actual")
            ax.plot(forecast_vals, label="Forecast", linestyle="--")
            ax.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.3)
            ax.set_title(f"{ticker} SARIMA Forecast")
            ax.legend()
            st.pyplot(fig)

            st.subheader("📋 Forecast Table")
            st.dataframe(forecast_df)
            download_csv(forecast_df, f"{ticker}_sarima_forecast.csv")

if __name__ == "__main__":
    main()
