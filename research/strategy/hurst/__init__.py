# Import main functions to make them available at the package level
from .core import (
    hurst_rs, 
    hurst_dfa, 
    hurst_aggvar, 
    hurst_higuchi,
    rolling_hurst
)
from .visualization import (
    plot_hurst_methods_comparison,
    plot_rolling_hurst
)
from .strategy import (
    hurst_trading_strategy,
    mean_reversion_strategy,
    trend_following_strategy
)
from .evaluation import (
    evaluate_strategy,
    save_strategy_results_to_csv
)
from .utils import calculate_atr