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


def  get_data(ticker):
    stock = yfi.get_data(ticker, start_date = 2022/0o4/0o1, end_date = None, index_as_date = True, interval = "1d")
    stock["date"] = pd.to_datetime(stock.index)
    stock.reset_index(inplace=True)
    stock_data = stock[["close",  "date"]]
    stock_data.columns = ['y', 'ds']
    return  stock_data

def get_name(ticker):
    company = yf.Ticker(ticker)
    company_name = company.info['longName']
    return company_name

def prophet(stock_data):
    m = Prophet(interval_width=0.95, daily_seasonality=True)
    model = m.fit(stock_data)
    future = m.make_future_dataframe(periods=100,freq='D')
    forecast = m.predict(future)
    return forecast, m

def model2(data):
    m = Prophet(daily_seasonality=True)
    # Train the model
    model = m.fit(data)
    future =  model.make_future_dataframe(periods=365)
    prediction = model.predict(future)
    return model,  prediction


