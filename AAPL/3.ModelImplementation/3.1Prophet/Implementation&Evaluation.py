import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv(r"C:\Users\shikh\Desktop\TIME SERIES STOCK PROJECT\AAPL\abcd\AAPL FROM SCRATCH\PROPHET\cleaned_data.csv")
df=df[['Date','Close']]  
df.columns= ['ds','y']  
df['ds'] =pd.to_datetime(df['ds'],dayfirst=True)

#Model loading
ProMod=Prophet()
ProMod.fit(df)
#Forecasting into 1 future year
future =ProMod.make_future_dataframe(periods=365)
forecast=ProMod.predict(future)
ProMod.plot(forecast)
plt.title("Forecasted close prices")
plt.show()
#Trends, Yearly ,Weekly, etc etc
ProMod.plot_components(forecast)
plt.show()

#Merge overlapping actual&predicted data
evaluate=df.merge(forecast[['ds','yhat']],on='ds',how='left')
print("Evaluation:")
mae=mean_absolute_error(evaluate['y'],evaluate['yhat'])
print(f"Mean Absolute Error={mae:.4f}")
rmse=mean_squared_error(evaluate['y'],evaluate['yhat'])**(0.5)
print(f"Root Mean Squared Error={rmse:.4f}")
r2=r2_score(evaluate['y'],evaluate['yhat'])
print(f"R²={r2:.4f}")
mape=(abs((evaluate['y']-evaluate['yhat'])/evaluate['y'])).mean()*100
print(f"MAPE={mape:.4f}%")
accuracy=100-mape
print(f"Accuracy(approx)={accuracy:.2f}%")

#CrossValidating
crossValid=cross_validation(ProMod, initial='730 day', period='180 day', horizon='365 day')
performance=performance_metrics(crossValid)
print("\n CrossValidation Metrics' summary:")
print(performance[['horizon','mae','rmse','mape']].head())

#Plots
fig=plot_cross_validation_metric(crossValid,metric='rmse')
plt.show()
#Merge actuals withforecast
merged=pd.merge(df,forecast[['ds','yhat']],on='ds',how='left')
plt.figure(figsize=(12,10))
#actual&forecast
plt.plot(df['ds'], df['y'],label='Original',color='black')
plt.plot(forecast['ds'],forecast['yhat'],label='Forecasted ',color='blue')
#forecast start here
plt.axvline(df['ds'].max(),color='red',linestyle=':',label='Forecast Begins From Here')
plt.title("Actualv/sForecasted Stock prices")
plt.xlabel("Date")
plt.ylabel("Closing price")
plt.legend()
plt.grid(True)
plt.show()

#Actual vs predicted for last 2 months
PastSixtyDays= evaluate.tail(60) 
plt.figure(figsize=(14,6))
plt.plot(PastSixtyDays['ds'],PastSixtyDays['y'],label='Original Data ',color='black')
plt.plot(PastSixtyDays['ds'],PastSixtyDays['yhat'],label='Forecasted data',color='blue')
plt.title("Actualv/sPredicted prices(Prev. 60 Days)")
plt.xlabel("Date")
plt.ylabel("Closing price")
plt.legend()
plt.grid(True)
plt.show()

forecast.to_csv("C:/Users/shikh/Desktop/TIME SERIES STOCK PROJECT/AAPL/AAPL_forecasted.csv", index=False)
print("Forecast Done.")# verifying whether the model is successfully working or not
