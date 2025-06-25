import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(ticker, start_date, end_date):
    """Download stock data using Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # If data has multi-level columns (e.g. ['Close']['AAPL']), flatten it
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data[['Close']]


def run_arima_forecast(series, p, d, q, forecast_steps):
    """Fit ARIMA model and generate forecasts."""
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    return forecast, model_fit

def auto_select_arima(series):
    """Automatically select best ARIMA parameters."""
    model = auto_arima(series, seasonal=False, suppress_warnings=True)
    return model.order