import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

Data=pd.read_csv(r"C:\Users\shikh\Desktop\TIME SERIES STOCK PROJECT\AAPL\abcd\AAPL FROM SCRATCH\EDA\CleanedD.csv", parse_dates=['Date'], index_col='Date')
#closing rates
plt.figure(figsize=(14,9))
Data['Close'].plot(title='AAPL Closing price Over Time')
plt.xlabel('Date ')
plt.ylabel('Closing  price')
plt.grid()
plt.show()
#moving averages
Data['MA30']=Data['Close'].rolling(window=30).mean()
Data['MA90']=Data['Close'].rolling(window=90).mean()
Data[['Close','MA30','MA90']].plot(figsize=(12,6),title='Moving Averages(30,90 Days)')
plt.grid()
plt.show()
#volatility
Data['RollingSTD']=Data['Close'].rolling(window=30).std()
Data['RollingSTD'].plot(figsize=(12,6),title='30-Day rolling Std-Devn')
plt.grid()
plt.show()
#seasonal decomp.
try:
    SD=seasonal_decompose(Data['Close'], model='additive', period=365)
    SDFig=SD.plot()
    SDFig.set_size_inches(14,11)
    plt.show()
except Exception as e:
    print(f"Seasonal decomp. failed:{e}")
