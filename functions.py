def  get_data(ticker, start_date):
    stock = yf.get_data(ticker, start_date = 2022/0o4/0o1, end_date = None, index_as_date = True, interval = "1d")
    stock["date"] = pd.to_datetime(stock.index)
    stock.reset_index(inplace=True)
    stock_price = stock[["close",  "date"]]
    stock_price.columns = ['y', 'ds']
    return stock_price

