import pandas as pd
import cProfile
import pstats
from src.strategy import MomentumStrategy
from src.backtester import BacktestEngine

def execute(capital: int = 100_000, lookback: int = 12, top_n: int = 20):
    strategy = MomentumStrategy()
    engine = BacktestEngine(
        strategy=strategy,
        start_date='2018-01-01',     
        cash=capital,                
        lookback=lookback,              
        top_n=top_n,                    
        skip_months=1               
    )
    
    results = engine.backtest()
    
    print()
    print(f"Initial Capital: ${capital:,}\n")
    print(f"Final Portfolio Value: ${results['history']['value'].iloc[-1]:,.2f}\n")
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
    print()

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    execute()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)