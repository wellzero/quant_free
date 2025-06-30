import sys


from quant_free.dataset.xq_daily_data import *
from quant_free.dataset.xq_symbol import *
from quant_free.factor.alpha_101 import *

from quant_free.utils.us_equity_utils import *
from quant_free.factor.sectors_ratio import *

def index_cal():

  start_date = get_json_config_value("start_date")
  end_date = get_json_config_value("end_date")

  market = "us"
  sector_file = 'equity_sector.csv'
  sectors = list(us_dir1_load_csv(market, dir0 = 'symbol', dir1 = 'xq', filename=sector_file)['name'].values)

  print(sectors)


  df_index = SectorsRatio(sectors, start_date, end_date, column_option = 'market_capital', dir = 'xq')
  result = df_index.ratio()
  us_dir1_store_csv(market, dir0 = 'symbol', dir1 = 'xq', filename='index_price' + '.csv', data = result[0])
  us_dir1_store_csv(market, dir0 = 'symbol', dir1 = 'xq', filename='index_price_ratio' + '.csv', data = result[1])

  df_index = SectorsRatio(sectors, start_date, end_date, column_option = 'amount', dir = 'xq')
  result = df_index.ratio()
  us_dir1_store_csv(market, dir0 = 'symbol', dir1 = 'xq', filename='index_amount' + '.csv', data = result[0])
  us_dir1_store_csv(market, dir0 = 'symbol', dir1 = 'xq', filename='index_amount_ratio' + '.csv', data = result[1])


  sector_file = 'equity_sector.csv'
  sectors = list(us_dir1_load_csv(market, dir0 = 'symbol', dir1 = 'fh', filename=sector_file)['Sector'].values)

  print(sectors)

  df_index = SectorsRatio(sectors, start_date, end_date, column_option = 'market_capital', dir = 'fh')
  result = df_index.ratio()
  us_dir1_store_csv(market, dir0 = 'symbol', dir1 = 'fh', filename='index_price' + '.csv', data = result[0])
  us_dir1_store_csv(market, dir0 = 'symbol', dir1 = 'fh', filename='index_price_ratio' + '.csv', data = result[1])

  df_index = SectorsRatio(sectors, start_date, end_date, column_option = 'amount', dir = 'fh')
  result = df_index.ratio()
  us_dir1_store_csv(market, dir0 = 'symbol', dir1 = 'fh', filename='index_amount' + '.csv', data = result[0])
  us_dir1_store_csv(market, dir0 = 'symbol', dir1 = 'fh', filename='index_amount_ratio' + '.csv', data = result[1])

if __name__ == "__main__":

  start_date = get_json_config_value("start_date")
  end_date = get_json_config_value("end_date")


  # index_cal()

  alpha_101_ = Alpha101(start_date, end_date, dir = 'xq')

  # symbol = 'AAPL'
  # tend = Alpha101(start_date, end_date, dir = 'xq')
  # df = tend.calc_1_symbol(symbol)
  # print(df.tail(10))




  sector_file = 'equity_sector.csv'
  sectors = list(us_dir1_load_csv(market = 'us', dir0 = 'symbol', dir1 = 'xq', filename=sector_file)['name'].values)

  sectors = ["电脑与外围设备"]
  sectors = ["半导体产品与设备"]
  
  df = alpha_101_.calc_sectors(sectors)
  