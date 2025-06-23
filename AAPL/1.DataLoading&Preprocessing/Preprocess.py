import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



df = yf.download('AAPL' , start="2015-01-01", end="2024-12-31",auto_adjust=False)
#print(df.head())

#df =df.dropna()  #remove any rows with missing values
# Step 2: Flatten MultiIndex columns
#df.columns = df.columns.get_level_values(0)

# Optional: Reset index if you want Date as column
df.reset_index(inplace=True)


data =df[['Close','High', 'Low']].copy() #extracting only closing price for forcasting

#data= data.reset_index() # moves the data from index to a column
#data.rename(columns={'Date':'Date', 'Close':'Close'}, inplace =True)



#plot the close price
plt .figure(figsize=(10,5))
plt.plot(df['Date'],df['Close'] , label ='Close Price')
plt. title(f"AAPL Closing Price")
plt.xlabel("Date");plt.ylabel("Price"); plt.legend();plt.tight_layout();plt.show()



#Data Scaling(for LSTM)

scaler = MinMaxScaler(feature_range=(0,1))
data_scaled= scaler.fit_transform(data)  # shape (n_samples, 1)




# Add the scaled values as a new column to the original DataFrame
scaled_df= pd.DataFrame(data_scaled,columns=['Closed_Scaled', 'High_Scaled', 'Low_Scaled']) 


# Step 7: Combine with original date and price columns and scaled data
df_combined = pd.concat(
    [df[['Date', 'Close', 'High', 'Low']], scaled_df],
    axis=1
)

# Step 8: Show the first few rows
print(df_combined.head())


