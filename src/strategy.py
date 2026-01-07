import pandas as pd

class MomentumStrategy():
    def __init__(self):
        self.momentum_scores_cache = None
        self.lookback_cache = None

    def run(self, current_date, monthly_returns, lookback, skip_months, top_n) -> pd.Series:
        # cache all momentum scores for the dataset
        if self.momentum_scores_cache is None or self.lookback_cache != lookback:
            self.momentum_scores_cache = monthly_returns.rolling(lookback).agg(lambda x : (x+1).prod()-1)
            self.lookback_cache = lookback

        # only look up what's necessary.
        momentum_scores = self.momentum_scores_cache.loc[:current_date]
        momentum_scores = momentum_scores.dropna(how='all')

        if len(momentum_scores) <= skip_months:
            # not enough data
            return {}

        curr = momentum_scores.iloc[-(skip_months + 1)]
        winners = curr.nlargest(top_n).index.tolist()
        target_weights = {ticker: 1.0 / top_n for ticker in winners}
        return target_weights
            