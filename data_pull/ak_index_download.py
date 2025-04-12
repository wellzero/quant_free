from pathlib import Path
import pandas as pd
from quant_free.dataset.us_equity_download import us_equity_daily_data_download
from quant_free.utils.us_equity_utils import create_common_directory

_this_dir = Path(__file__).parent.parent

def ak_index_download(market='us', symbols=None, provider='yfinance'):
    """
    Download index daily trade data from specified provider.
    
    Args:
        market: Market identifier (default 'us')
        symbols: List of index symbols to download (default None downloads all)
        provider: Data provider ('yfinance' or 'xq')
    """
    if symbols is None:
        # Default to common US indices
        symbols = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']
    
    if provider == 'xq':
        from quant_free.dataset.us_equity_xq_download import equity_xq_daily_data_download
        return equity_xq_daily_data_download(market=market, symbols=symbols)
    else:
        return us_equity_daily_data_download(market=market, symbols=symbols, provider=provider)

if __name__ == '__main__':
    # Example usage
    ak_index_download()
