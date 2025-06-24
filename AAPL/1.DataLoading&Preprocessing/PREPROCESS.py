import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller

raww=yf.download("AAPL" ,start="2015-01-01",end="2024-12-31",auto_adjust=False)
raww.to_csv("rawD.csv")
#CLEAN
Data=raww.copy()
Data.reset_index(inplace=True)
Data.dropna(inplace=True)
Data.drop_duplicates(inplace=True)
Data['Date']=pd.to_datetime(Data['Date'])
Data.sort_values('Date',inplace=True)
Data.set_index('Date',inplace=True)
CleanedD=Data[['Close','High','Low']]  
CleanedD.to_csv("CleanedD.csv")
#NORMALISING
minmaxScaler=MinMaxScaler(feature_range=(0,1))
normalized=pd.DataFrame(minmaxScaler.fit_transform(CleanedD),
                          columns=['Close_Norm','High_Norm','Low_Norm'],
                          index=CleanedD.index)
normalized.to_csv("normalizedD.csv")
#STANDARDIZING
standardScaler=StandardScaler()
standardized=pd.DataFrame(standardScaler.fit_transform(CleanedD),
                            columns=['Close_Std','High_Std','Low_Std'],
                            index=CleanedD.index)
standardized.to_csv("standardizedD.csv")
#DETECTING STATIONARITY
print("\nSTATIONARY TEST RESULTS:")
for col in CleanedD.columns:
    result=adfuller(CleanedD[col])
    print(f"\n {col}")
    print(f"Adfuller Statistic:{result[0]:.4f}")
    print(f"p-value:{result[1]:.4f}")
    if result[1]<0.05:
        print("This Stock is Stationary.")
    if result[1]>=0.05:
        print("This Stock is Non-Stationary.")
#first few rows
print(normalized.head())
