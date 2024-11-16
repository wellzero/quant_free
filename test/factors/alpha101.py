import sys
from pathlib import Path
_this_dir = Path(__file__).parent.parent.parent
sys.path.append(str(_this_dir))

from quant_free.dataset.us_equity_load import *
from quant_free.factor.alpha_101 import *

if __name__ == "__main__":

  start_date = get_json_config_value("start_date")
  end_date = get_json_config_value("end_date")


  alpha_101_ = Alpha101(start_date, end_date, dir = 'xq')
  df = alpha_101_.calc('互联网与直销零售')
