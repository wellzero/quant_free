from quant_free.finml.labeling.labeling import *
from quant_free.finml.features.volatility import daily_volatility

def meta_labeling_getting(data,
                  num_days = 7,
                  lookback = 60,
                  pt_sl = [2, 1],
                  min_ret = 0.01, # minimum position return
                  num_threads = 1, # number of multi-thread 
                  ):
  vertical_barrier = add_vertical_barrier(
      data.index, 
      data['close'], 
      num_days = num_days # expariation limit
      )
  volatility = daily_volatility(
      data['close'], 
      lookback = lookback # moving average span
      )
  triple_barrier_events = get_events(
      close = data['close'],
      t_events = data.index[2:],
      pt_sl = pt_sl, # profit taking 2, stopping loss 1
      target = volatility, # dynamic threshold
      min_ret = min_ret, # minimum position return
      num_threads = num_threads, # number of multi-thread 
      vertical_barrier_times = vertical_barrier, # add vertical barrier
      side_prediction = None # betting side prediction (primary model)
      )
  labels = meta_labeling(
      triple_barrier_events, 
      data['close']
      )
  triple_barrier_events['side'] = labels['bin']

  meta_labels = meta_labeling(
      triple_barrier_events, # with side labels
      data['close']
      )

  return meta_labels