
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Load your standardized series
df = pd.read_csv("standardized_data.csv", index_col="Date", parse_dates=True)
ts = df["Close_Std"]

# 2. Quick visualization of the last 100 points
ts[-100:].plot(title="AAPL Standardized Close (Last 100 Days)", figsize=(12, 5))
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. ACF & PACF to choose (p, q)
plot_acf(ts.dropna(), lags=30)
plot_pacf(ts.dropna(), lags=30)
plt.show()

# 4. Specify & fit your ARIMA(p,d,q)
p ,d,q= 2,1,1
print(f"\nTraining ARIMA({p},{d},{q}) model...")
model = ARIMA(ts, order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())

# 5. Forecast next 30 business days
forecast_steps = 30
forecast_vals = model_fit.forecast(steps=forecast_steps)
forecast_idx = pd.date_range(start=ts.index[-1],
                             periods=forecast_steps + 1,
                             freq="B")[1:]
forecast_series = pd.Series(forecast_vals, index=forecast_idx, name="ARIMA_Forecast")

# 6. Plot historical vs. forecast
plt.figure(figsize=(12, 6))
plt.plot(ts[-100:], label="Historical")
plt.plot(forecast_series, label="Forecast", linestyle="--")
plt.title(f"ARIMA({p},{d},{q}) — Next {forecast_steps} Business Days")
plt.xlabel("Date")
plt.ylabel("Standardized Close")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Export forecast to CSV
forecast_series.to_csv("arima_forecast.csv", header=True)
print("Forecast saved to arima_forecast.csv")

# 8. (Optional) Evaluate with RMSE on last 30 points
train, test = ts[:-forecast_steps], ts[-forecast_steps:]
eval_model = ARIMA(train, order=(p, d, q)).fit()
pred = eval_model.forecast(steps=forecast_steps)
rmse = np.sqrt(mean_squared_error(test, pred))
print(f"RMSE on last {forecast_steps} days: {rmse:.4f}")
