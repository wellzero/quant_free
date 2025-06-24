import argparse
from typing import List
from quant_free.dataset.xq_finance_parser import * 

def process_market(markets: List[str]) -> None:
    """Process data for specific markets."""
    for market in markets:
        print(f"Processing {market} market...")
        xq_finance_parser(market=market)

def process_symbols(symbols: List[str], market: str) -> None:
    """Process specific symbols in a market."""
    for symbol in symbols:
        print(f"Processing {symbol} in {market} market...")
        xq_finance_data(market=market, symbol=symbol)

def main() -> None:
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Process XQ Finance data.")
    parser.add_argument('--markets', default='us,cn,hk',
                       help='Comma-separated list of markets (default: us,cn,hk)')
    parser.add_argument('--symbols', default=None,
                       help='Comma-separated list of symbols to process')
    # parser.add_argument('--symbol-market', default=None,
    #                    help='Market for symbols (required if --symbols is used)')
    
    args = parser.parse_args()

    if args.symbols:
        # if not args.symbol_market:
        #     raise ValueError("--symbol-market is required when using --symbols")
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        process_symbols(symbols, args.markets.lower())
    else:
        markets = [m.strip().lower() for m in args.markets.split(',')]
        process_market(markets)

if __name__ == "__main__":
    main()
