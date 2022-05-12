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

# Functions to work with in this project:
from functions import get_data, get_name, prophet
# Streamlit web.
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

with dataset:
    st.header("Data Acquisition")  
    st.write("In this project I will use the data stored in  YahooFinance Website with Yahoo! Finance's API")
    st.write('[https://finance.yahoo.com/](https://finance.yahoo.com/)')              
    st.markdown("""---""")
  
company = st.selectbox('Please select company ticker:',
                                  ('None', "PEP", "MSFT", "TSLA", "AMZN", "BRK.B", "XOM",  "BAC"))      
st.markdown("""---""")
st.write('\n')
st.spinner(text="In progress...")
with st.spinner('Wait for it...'):
    time.sleep(9)
st.write('\n') 

#st.write(company_name)
col1, col_mid, col2 = st.columns((1, 0.1, 1))
with col1:
    company_df = get_data(company)
    company_name = get_name(company)
    st.write(company_name)
    st.write(company_df.head())

with col2:
    #Funtion to display company website or company logo. 
    st.write('[https://www.tesla.com/](https://www.tesla.com/)')
    st.write(f'<iframe \
                     width="400" \
                     height="300"\
                     src="https://en.wikipedia.org/wiki/Tesla,_Inc."></iframe>',
                     unsafe_allow_html=True )   

st.markdown("""---""")
st.write('\n')

with modelTraining:
    st.write("In this step I will train the model with historic stock price data to predict future prices")
    st.write(prophet(company_df))
