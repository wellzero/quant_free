import akshare

def download_index_daily_trade_data():
    index_data = akshare.index_daily(symbol="000001")  # Example symbol
    print(index_data)

download_index_daily_trade_data()


