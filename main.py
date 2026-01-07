import pandas as pd
import cProfile
import pstats
from src.strategy import MomentumStrategy
from src.backtester import BacktestEngine

def execute():
    initial_capital = 100_000
    lookback = 12
    top_n = 20
    strategy = MomentumStrategy()
    engine = BacktestEngine(
        strategy=strategy,
        start_date='2018-01-01',     
        cash=initial_capital,                
        lookback=lookback,              
        top_n=top_n,                    
        skip_months=1               
    )
    results = engine.backtest()
    print(f"\nFinal Portfolio Value: ${results['history']['value'].iloc[-1]:,.2f}\n")
    print(f"Initial Capital: ${initial_capital:,}\n")
    print("Metrics:")
    for key, value in results['metrics'].items():
        if key == 'num_periods':
            print(f"  {key}: {value}")
        elif key == 'sharpe_ratio':
            print(f"  {key}: {value:.2f}")
        elif pd.isna(value) or value is None:
            print(f"  {key}: N/A")
        else:
            print(f"  {key}: {value:.2%}")
    print()
    history = results['history']
    engine.plot_results(portfolio_history=history)
    print()
    
    print("============= Profile Results =============")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    execute()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)