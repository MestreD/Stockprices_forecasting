#Libraries to work with
import pandas as pd
import warnings; 
warnings.simplefilter('ignore')
#Yahoo Finance API
import yahoo_fin.stock_info as yf
tickers = yf.tickers_sp500()
# Prophet
from prophet import Prophet
from prophet.plot import plot_plotly
# Interactive plots
import plotly.io as pio
pio.renderers.default = "notebook_connected"

stock = yf.get_data("aapl", start_date = 2022/0o4/0o1, end_date = None, index_as_date = True, interval = "1d")

# Functions to work with in this project:
from functions import get_data
# Streamlit web.
st.set_page_config(page_title="Stock Price Forecasting",
        page_icon="chart_with_upwards_trend", layout="wide")

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()


with header:
    st.title("Forecasting model to predict future stock prices.")
    st.write("This project is an attempt to build a model that predicts the future prices of a chosen public company from the S&P 500.")
    st.write('\n')
    


with dataset:
    st.header("Data Acquisition")  
    st.write("Download market data from Yahoo! Finance's API")
    st.markdown("""---""")
  
company = st.selectbox('Please select company ticker:',
                                  ('None', "PEP", "MSFT", "TSLA", "AMZN", "BRK.B", "XOM",  "BAC"))    
st.write('\n')
st.write('\n') 

if company == "TSLA":
        company_name = "Tesla"
        st.write(company_name)
        companyA_df = scrape("TSLA", 100)
        st.write(company_df.head())