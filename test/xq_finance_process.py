import argparse
from quant_free.dataset.xq_finance_data import xq_finance_process

def process_market(market: str) -> None:
    """Process data for a specific market."""
    print(f"Processing {market} market..."  )
    for market in market:
        xq_finance_process(market=market)


def main() -> None:
    """Main function to handle command-line arguments and process markets."""
    parser = argparse.ArgumentParser(description="Process XQ Finance data for different markets.")
    parser.add_argument('--markets', default='us,cn,hk',
                       help='Comma-separated list of markets to download (default: us,cn,hk)')
    
    args = parser.parse_args()

    markets = [m.strip().lower() for m in args.markets.split(',')]

    process_market(markets)


if __name__ == "__main__":
    main()
