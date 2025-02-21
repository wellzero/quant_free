import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent.parent
sys.path.append(str(_this_dir))

from quant_free.dataset.us_equity_load import *
from quant_free.factor.trend import *

if __name__ == "__main__":

  start_date = get_json_config_value("start_date")
  end_date = get_json_config_value("end_date")

  # symbol = 'AAPL'
  # tend = Trend(start_date, end_date, dir = 'xq')
  # df = tend.calc_1_symbol(symbol)
  # print(df.tail(10))

  # sector = "半导体产品与设备"
  # sector = "电脑与外围设备"

  trend = Trend(start_date, end_date, dir = 'xq')
  
  # df = trend.calc_1_sector(sector)
  
  # sectors = ["互联网与直销零售"]

  sector_file = 'us_equity_sector.csv'
  sectors = list(us_dir1_load_csv(dir0 = 'symbol', dir1 = 'xq', filename=sector_file)['name'].values)

  # sectors = ["互联网与直销零售"]
  sectors = ["半导体产品与设备", "电脑与外围设备"]

  df = trend.calc_sectors(sectors)
  # df = trend.parallel_calc_sectors(sectors)