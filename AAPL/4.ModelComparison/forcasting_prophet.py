import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

st.set_page_config(layout="wide",page_title="Prophetforecast")
st.title("Stock-Forecasting using Prophet")
#sidebar
with st.sidebar:
    st.header("Upload & Config")
    data=st.file_uploader("Upload cleaned CSV (Date, Close)",type=["csv"])
    forecastDur=st.slider("Forecast Horizon (days)",30,730,365,step=30)
    tailDur=st.slider("Actual vs Predicted: Last N Days",30,180,60,step=10)
    st.markdown("---")
    st.subheader("Cross-Validation Settings")
    initialTrain=st.slider("Initial Training Window",365,1065,730,step=30)
    EvalPeriod=st.slider("Evaluation Period",90,365,180,step=30)
    HorizonPeriod=st.slider("Prediction Horizon",90,365,365,step=30)
if not data:
    st.warning("Please upload a cleaned CSV with 'Date' and 'Close'.")
    st.stop()
#loadig data and fitting model
df=pd.read_csv(data)
df=df[['Date','Close']].copy()
df.columns=['ds','y']
df['ds']=pd.to_datetime(df['ds'],dayfirst=True)
ProMod=Prophet()
ProMod.fit(df)
#predicting
future=ProMod.make_future_dataframe(periods=forecastDur)
forecast=ProMod.predict(future)
#evaluating
st.subheader("Evaluation Metrics")
col1, col2, col3, col4, col5=st.columns(5)
evaluate=df.merge(forecast[['ds','yhat']],on='ds',how='left')
mae=mean_absolute_error(evaluate['y'],evaluate['yhat'])
col1.metric("MAE",f"{mae:.2f}")
rmse=mean_squared_error(evaluate['y'],evaluate['yhat'])**(0.5)
col2.metric("RMSE",f"{rmse:.2f}")
r2=r2_score(evaluate['y'],evaluate['yhat'])
col3.metric("R²", f"{r2:.4f}")
mape=(abs((evaluate['y']-evaluate['yhat'])/evaluate['y'])).mean()*100
col4.metric("MAPE",f"{mape:.2f}%")
accuracy=100-mape
col5.metric("Accuracy",f"{accuracy:.2f}%")
#predcted plot
st.subheader("Forecasted Closing Prices")
with st.expander("Show Forecast Plot"):
    fig1=ProMod.plot(forecast)
    plt.title(f"Forecasted Close Prices ({forecastDur} days ahead)")
    st.pyplot(fig1)
#trends, etc
st.subheader("Trend&Seasonality")
with st.expander("Show Trend&Seasonality Components"):
    fig2=ProMod.plot_components(forecast)
    st.pyplot(fig2)
#originalvs forcasted
merged=pd.merge(df, forecast[['ds','yhat']],on='ds',how='left')
st.subheader("Actual vs Forecasted(Full Timeline)")
with st.expander("Show Actual vs forecast Plot"):
    fig3,ax3=plt.subplots(figsize=(12, 6))
    ax3.plot(df['ds'],df['y'],label='Original',color='black')
    ax3.plot(forecast['ds'],forecast['yhat'],label='Forecasted',color='blue')
    ax3.axvline(df['ds'].max(),color='red',linestyle=':',label='Forecast Begins')
    ax3.set_title("Actual v/s Forecasted Stock Prices")
    ax3.set_xlabel("Date"); ax3.set_ylabel("Price")
    ax3.legend(); ax3.grid(True)
    st.pyplot(fig3)
#prev days plot 
st.subheader(f"Prev {tailDur} Days Actualv/sPredicted")
with st.expander("Show Last-N-Days Comparison"):
    PastSixtyDays=evaluate.tail(tailDur)
    fig4, ax4=plt.subplots(figsize=(12, 6))
    ax4.plot(PastSixtyDays['ds'],PastSixtyDays['y'],label='Original Data',color='black')
    ax4.plot(PastSixtyDays['ds'],PastSixtyDays['yhat'],label='Forecasted Data',color='blue')
    ax4.set_title(f"Actualv/sPredicted (Last {tailDur} Days)")
    ax4.set_xlabel("Date"); ax4.set_ylabel("Price")
    ax4.legend(); ax4.grid(True)
    st.pyplot(fig4)
#crossValidate
st.subheader("Prophet Cross-Validation Results")
with st.spinner("Cross-Validating:-"):
    crossValid=cross_validation(
        ProMod,
        initial=f'{initialTrain} days',
        period=f'{EvalPeriod} days',
        horizon=f'{HorizonPeriod} days'
    )
    performance=performance_metrics(crossValid)
    st.dataframe(performance[['horizon','mae','rmse','mape']].round(2))
    fig_cv=plot_cross_validation_metric(crossValid,metric='rmse')
    st.pyplot(fig_cv)
#download 
st.download_button(
    "Download forecast(in csv)",
   forecast.to_csv(index=False).encode('utf-8'),
   file_name="forecast.csv",
    mime="text/csv"
)
st.success("Forecasted successfully.")# verifying that the app has successfully worked! lol