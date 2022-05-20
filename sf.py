#Libraries to work with
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import warnings; 
warnings.simplefilter('ignore')

#Yahoo Finance API´s
import yahoo_fin.stock_info as yfi
tickers = yfi.tickers_sp500()
import yfinance as yf
# Prophet
from prophet import Prophet
from prophet.plot import plot_plotly
# Interactive plots
import plotly.io as pio
pio.renderers.default = "notebook_connected"
import time
from PIL import Image
#opening the image
image = Image.open('yahoo_finance_logo.png')
import datetime

# Functions to work with in this project:
#from functions import get_data, get_name, prophet, model2
@st.cache 
def  get_data(ticker):
    stock = yfi.get_data(ticker, start_date = '01/01/1999', end_date = None, index_as_date = True, interval = "1d")
    stock["date"] = pd.to_datetime(stock.index)
    stock.reset_index(inplace=True)
    stock_data = stock[["close",  "date"]]
    stock_data.columns = ['y', 'ds']
    return  stock_data
@st.cache 
def get_name(ticker):
    company = yf.Ticker(ticker)
    company_name = company.info['longName']
    return company_name
@st.cache
def prophet(stock_data):
    m = Prophet(interval_width=0.95, daily_seasonality=True)
    model = m.fit(stock_data)
    future = m.make_future_dataframe(periods=100,freq='D')
    forecast = m.predict(future)
    return forecast, m
@st.cache
def prophet2(data):
    m = Prophet(daily_seasonality=True)
    # Train the model
    model = m.fit(data)
    future =  model.make_future_dataframe(periods=365)
    prediction = model.predict(future)
    return model,  prediction

# Streamlit web.
st.set_page_config(page_title="Stock Price Forecasting",
        page_icon="chart_with_upwards_trend", layout="wide")

header = st.container()
dataset = st.container()
features = st.container()

with header:
    st.title("Forecasting model to predict future stock prices.")
    st.write("This project is an attempt to build a model that predicts the future prices of a chosen public company from the S&P 500, using the  library Prophet to forecast time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.")
    st.write('\n')
    st.markdown("""---""")

with dataset:
    col1, col_mid, col2 = st.columns((1, 0.1, 1))
    with col1:    
        st.header("Data Acquisition")  
        st.write("In this project I will use the data stored in  YahooFinance Website with Yahoo! Finance's API")
        st.write('[Documentation](http://theautomatic.net/yahoo_fin-documentation/)')   
    with col2:
#displaying the image on streamlit app
        st.image(image)
        st.write('[https://finance.yahoo.com/](https://finance.yahoo.com/)')   
    st.markdown("""---""") 

with features:
    company = st.selectbox('Please select company ticker:',
                                    ('None', "PEP", "MSFT", "TSLA", "AMZN", "BRK.B", "XOM",  "BAC"))    
    if company == "None":
        st.write('You must select a ticker')  
    else:
        st.write('\n')
        st.spinner(text="In progress...")
        with st.spinner('Wait until the running icon on the top right stops. It might take a couple of minutes...'):
            time.sleep(30)
        st.write('\n') 
        
        col1, col_mid, col2 = st.columns((1, 0.1, 1))
        with col1:
            stock_data = get_data(company)
            company_name = get_name(company)

            st.write(company_name)
            st.dataframe(stock_data)

        with col2:
            st.video("https://www.youtube.com/watch?v=6Af5GUqh_Kk&ab_channel=BenHider")
            st.write('\n')
        st.markdown("""---""")
        st.header("Forecasting")  
        st.write("In this step I will train the model with the historic stock price data collected above to predict future prices.")
        st.write("In the below graph you´ll see the black dots representing the data given to the model, the blue line or \"yhat\"represents the prediction and the \"yhat_lower, yhat_upper\" represents the uncertainty intervals.")
        
        forecast, m = prophet(stock_data)
        st.dataframe(forecast.tail())
        st.markdown("""---""")
        st.write(m.plot(forecast))
        st.write(m.plot_components(forecast))
        st.markdown("""---""")
        st.subheader("Lest test how close the prediction is!")
        last_prices = stock_data[len(stock_data)-20:]
        data = stock_data[:-20]
        
        model, prediction = prophet2(data)
        d = st.date_input("Select a date to compare the actual closing price against the model prediction price for that day (Remember on weekends and holidays the market was closed):", datetime.date(2022,5,3))
        st.spinner(text="In progress...")
        with st.spinner('Wait until the running icon on the top right stops. It might take a couple of minutes...'):
            time.sleep(30)
        d = str(d)
        prediction_day = prediction[prediction.ds == d]["yhat"].values
        real = last_prices[last_prices.ds == d]["y"].values
        col1, col_mid, col2 = st.columns((1, 0.1, 1))
        with col1:
            st.write('The prediction price is:',  prediction_day)
        with col2:
            st.write('The real price is:',  real)
        if len(real) > 0:
            st.markdown("""---""")
            change_percent = "{:.1%}".format(((float(prediction_day[0])-real[0])/real[0]))
            st.write("The predition wasn't to accurate. The predicted price has a " + str(change_percent) + " difference from the actual price so we could say that we could miss that profit or exit the market too soon in an hypothetical investment.  The futures projects to work with will be complemented with  different timeseries libraries to compare results and find the most accurate")
            st.subheader("Prediction Visualization")
            st.write(plot_plotly(model, prediction))
            st.subheader("And that brings us to the end. ...Thank you so much for your interest.")
        if len(real) < 1:
            st.write("Select a valid date.")




    

    

