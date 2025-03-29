from quant_free.dataset.us_equity_load import *

class SectorsRatio:
  def __init__(self, sectors, start_date, end_date, column_option = 'market_capital', dir = 'fh', market = 'us'):
     self.sectors = sectors
     self.start_date = start_date
     self.end_date = end_date
     self.dir = dir
     self.column_option = column_option
     self.market = market

  def calc(self):
      dict_index = {}
      for sector in self.sectors:
        print(f"processing {sector} ...")
        data_symbols = us_dir1_load_csv(self.market, dir0 = 'symbol', dir1 = self.dir, filename= sector +'.csv')
        if (data_symbols.empty == False):
          symbols = data_symbols['symbol'].values

          data = equity_daily_data_load_within_range(
                                            self.market, 
                                            symbols = symbols, 
                                            start_date = self.start_date,
                                            end_date = self.end_date, 
                                            column_option = self.column_option, 
                                            dir_option = 'xq')
          if (len(data) > 0):
            df = pd.DataFrame(data)
            df_sum = df.sum(axis=1)
            index = df_sum #* 1000 /df_sum.iloc[0]
            dict_index[sector] = index
          else:
            self.sectors.remove(sector)
        else:
           print(f"remove {sector} ...")
           self.sectors.remove(sector)

      df = pd.DataFrame(dict_index)
      return df

  def ratio(self, days = 1):
     df = self.calc()
    #  print(df)
    #  df = df.fillna(0)
     return [df, df.pct_change(days).round(5)]