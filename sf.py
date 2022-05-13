#Libraries to work with
import streamlit as st
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
image = Image.open('yahooweb.png')
import datetime

# Functions to work with in this project:
from functions import get_data, get_name, prophet, model2

# Streamlit web.

base="dark"
primaryColor="purple"
st.set_page_config(page_title="Stock Price Forecasting",
        page_icon="chart_with_upwards_trend", layout="wide")

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()


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
        st.warning('You must select a ticker')  
    else:
        st.write('\n')
        st.spinner(text="In progress...")
        with st.spinner('Wait for it...'):
            time.sleep(9)
        st.write('\n') 
        
        col1, col_mid, col2 = st.columns((1, 0.1, 1))
        with col1:
            stock_data = get_data(company)
            company_name = get_name(company)
            st.write(company_name)
            st.write(stock_data.tail())
        with col2:
            #Funtion to display company website or company logo. 
            st.video("https://www.youtube.com/watch?v=6Af5GUqh_Kk&ab_channel=BenHider")
    st.write('\n')
    st.markdown("""---""")

    with modelTraining:
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
        model, forecast = model2(data)
        st.write(plot_plotly(model, forecast))
        d = st.date_input("Select a date to compare the actual closing price against the model prediction price", datetime.date(2019, 7, 6))
        st.write('The prediction price is:', forecast[forecast.ds == d]["yhat"])
        st.write('The real price is:', last_prices[last_prices.ds == d]["y"])
        st.markdown("""---""")
        st.write("Maybe the predition wasn't to accurate.. but next time ill be using different timeseries libraries and tune they're parameters  to compare results.")






    

    

