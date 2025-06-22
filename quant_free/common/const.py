# This file is part of QuantFree, a free and open-source quantitative trading platform.
from enum import Enum

# --- xueqiu finance constants ---
class Market(Enum):
    US = 'us'
    HK = 'hk'
    CN = 'cn'

MARKET_IGNORE_COLUMNS = {
    Market.US.value: 6,
    Market.HK.value: 5,
    Market.CN.value: 3,
}