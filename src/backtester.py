import numpy as np
import pandas as pd
import yfinance as yf
import ssl
import matplotlib.pyplot as plt
import os

class DataHandler():
    def __init__(self, start_date):
        self.start_date = start_date
        
    def get_tickers(self):
        # add certificate to permit wikipedia scrapihng
        ssl._create_default_https_context = ssl._create_unverified_context
        
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # can prob change this header depending on your device, a bit hacky here
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'} 
        
        tickers = pd.read_html(url, storage_options=headers)[0]['Symbol']
        df_ = pd.read_html(url, storage_options=headers)[1]
        
        # clean up data, remove survivorship bias, deduplicate
        df_['Effective Date'] = pd.to_datetime(df_['Effective Date'].stack()).unstack()
        df_ = df_[df_['Effective Date']["Effective Date"] >= self.start_date]
        tickers = tickers[~(tickers.isin(df_.Added.Ticker))]
        
        tickers_rem = df_.Removed.Ticker
        tickers = pd.concat([tickers, tickers_rem])
        
        tickers.drop_duplicates(inplace=True)
        tickers.dropna(inplace=True)
        # start fetching data for each symbol
        return tickers
    
    def load_data(self):
        tickers = self.get_tickers()
        prices, symbols = [], []
        for symbol in tickers:
            df = yf.download(symbol, start=self.start_date, progress=False)["Close"]
            if not df.empty:
                # ensure we're working with a series
                if isinstance(df, pd.DataFrame):
                    df = df.squeeze()

                if not isinstance(df, pd.Series):
                    continue

                missing_count = df.isna().sum()
                missing_pct = missing_count / len(df)

                if missing_pct < 0.05 and len(df) >= 252:  # filter inadequate data, can benefit from additional tuning
                    prices.append(df)
                    symbols.append(symbol)
                
        # raw price data for execution
        all_prices = pd.concat(prices, axis=1)
        all_prices.columns = symbols

        #generate monthly returns for strategy calculation
        monthly_returns = all_prices.pct_change(fill_method=None).resample("ME").agg(lambda x : (x + 1).prod()-1)

        valid_tickers = []
        for ticker in monthly_returns.columns:
            missing_months = monthly_returns[ticker].isna().sum()
            total_months = len(monthly_returns)
            missing_pct_monthly = missing_months / total_months
            
            if missing_pct_monthly < 0.05:
                valid_tickers.append(ticker)
            else:
                print(f"Filtering out {ticker}: {missing_pct_monthly:.1%} of monthly data missing")

        # Keep only valid tickers
        all_prices = all_prices[valid_tickers]
        monthly_returns = monthly_returns[valid_tickers]
        print()
        print("Data Summary:")
        print(f"  Total tickers after filtering: {len(valid_tickers)}")
        print(f"  Date range: {all_prices.index[0]} to {all_prices.index[-1]}")
        print(f"  Monthly returns range: {monthly_returns.index[0]} to {monthly_returns.index[-1]}")
        print(f"  Total months: {len(monthly_returns)}\n")
        return all_prices, monthly_returns
    
class Broker():
    def __init__(self, investment: int, price_data: pd.DataFrame):
        self.investment = investment
        self.total_cash = investment
        self.price_data = price_data
        self.holdings = {}
        self.preprocess_price_data()

    def preprocess_price_data(self): # TODO: Consolidate validate_prices into this to form one validation function
        # initial preprocessing to remove dates with NaN prices
        valid_cols = self.price_data.columns[~self.price_data.isna().any()]
        removed_tickers = set(self.price_data.columns) - set(valid_cols)

        if len(removed_tickers) > 0:
            print(f"Preprocessing: Removed {len(removed_tickers)} tickers with NaN prices")
            self.price_data = self.price_data[valid_cols]
        initial_rows = len(self.price_data)
        self.price_data = self.price_data.dropna(axis=0, how='any')
        removed_rows = initial_rows - len(self.price_data)

        if removed_rows > 0:
            print(f"Preprocessing: Removed {removed_rows} dates with NaN prices")

    def validate_prices(self, date, tickers):
        # extra layer of validation
        if isinstance(tickers, dict):
            valid_tickers = {}
            for ticker, value in tickers.items():
                if ticker not in self.price_data.columns:
                    continue
                if pd.isna(self.price_data.loc[date, ticker]):
                    continue
                valid_tickers[ticker] = value
            if len(valid_tickers) < len(tickers):
                removed = set(tickers.keys()) - set(valid_tickers.keys())
                # print(f"Filtered out {len(removed)} tickers with invalid prices on {date}: {removed}")
            return valid_tickers
        else:
            valid_tickers = []
            for ticker in tickers:
                if ticker not in self.price_data.columns:
                    continue
                if pd.isna(self.price_data.loc[date, ticker]):
                    continue
                valid_tickers.append(ticker)
            if len(valid_tickers) < len(tickers):
                removed = set(tickers) - set(valid_tickers)
                print(f"Warning: Filtered out {len(removed)} tickers with invalid prices on {date}: {removed}")

            return valid_tickers

    def get_portfolio(self) -> dict:
        # return current portfolio & holdings
        return {
            'holdings': self.holdings,
            'cash': self.total_cash
        }
        
    def get_portfolio_value(self, date):
        value = self.total_cash
        for ticker, shares in self.holdings.items():
            price = self.price_data.loc[date, ticker]
            value += shares * price
        return value
        
    def execute(self, target_weights, date):
        target_weights = self.validate_prices(date, target_weights)

        if not target_weights:
            print(f"No valid tickers to trade on {date}")
            return

        current_value = self.get_portfolio_value(date)

        # liquidate positions not in target_weights
        for ticker in list(self.holdings.keys()):
            if ticker not in target_weights:
                shares = self.holdings[ticker]
                # Validate price is available for liquidation
                if ticker in self.price_data.columns and not pd.isna(self.price_data.loc[date, ticker]):
                    price = self.price_data.loc[date, ticker]
                    proceeds = shares * price
                    self.total_cash += proceeds
                    del self.holdings[ticker]
                else:
                    print(f"Warning: Cannot liquidate {ticker} on {date}, no valid price")

        for ticker, weight in target_weights.items():
            target_dollar_amount = current_value * weight
            current_price = self.price_data.loc[date, ticker]
            target_shares = target_dollar_amount / current_price

            current_shares = self.holdings.get(ticker, 0)
            shares_to_trade = target_shares - current_shares

            if shares_to_trade > 0:  # buy
                cost = shares_to_trade * current_price
                self.total_cash -= cost
                self.holdings[ticker] = target_shares
            elif shares_to_trade < 0:  # sell
                proceeds = abs(shares_to_trade) * current_price
                self.total_cash += proceeds
                self.holdings[ticker] = target_shares

    
class BacktestEngine():
    def __init__(
        self, 
        strategy, 
        start_date, 
        cash:int = 10_000, 
        lookback:int = 12, 
        top_n:int = 10, 
        skip_months:int = 1
        ):
        self.strategy = strategy
        self.lookback = lookback
        self.top_n = top_n
        self.skip_months = skip_months
        data_handler = DataHandler(start_date=start_date)
        price_data, self.monthly_returns = data_handler.load_data()
        self.broker = Broker(investment=cash, price_data=price_data)
    
    def calculate_metrics(self, portfolio_history):
        returns = portfolio_history['value'].pct_change().dropna()
        total_return = (portfolio_history['value'].iloc[-1] / portfolio_history['value'].iloc[0]) - 1
        
        # annualized return
        num_periods = len(portfolio_history)
        years = num_periods / 12
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # sharpe ratio 
        mean_return = returns.mean()
        std_return = returns.std()
        annualized_mean = mean_return * 12
        annualized_std = std_return * np.sqrt(12)
        rate_free_returns = 0.027 # NOTE: I'm using a fixed rate here rather than a daily time-series rate for simplicity.
        sharpe_ratio = (annualized_mean - rate_free_returns) / annualized_std if annualized_std > 0 else 0
        
        # volatility
        volatility = std_return * np.sqrt(12)
        
        # maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'num_periods': num_periods
        }

        
    def backtest(self):
        # only use dates that exist in both monthly_returns and price_data
        available_dates = self.monthly_returns.index.intersection(self.broker.price_data.index)
        dates = available_dates[self.lookback + self.skip_months:]

        if len(dates) == 0:
            raise ValueError("No valid dates available for backtesting after warmup period")

        portfolio_history = []
        for date in dates:
            # iterate through signals at each date/weight instance
            target_weights = self.strategy.run(
                current_date=date,
                monthly_returns=self.monthly_returns,
                lookback=self.lookback,
                skip_months=self.skip_months,
                top_n = self.top_n
                )
            if not target_weights:
                continue
            self.broker.execute(target_weights=target_weights, date=date)
            portfolio_value = self.broker.get_portfolio_value(date=date)
            portfolio_history.append({
                'date': date,
                'value': portfolio_value
            })

        if len(portfolio_history) == 0:
            raise ValueError("No portfolio history generated - check data and parameters")

        portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
        metrics = self.calculate_metrics(portfolio_df)
        return {
            'portfolio': self.broker.get_portfolio(),
            'metrics': metrics,
            'history': portfolio_df
        }

    def fetch_sp500(self, start_date, end_date):
        sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)['Close']
        if isinstance(sp500, pd.DataFrame):
            sp500 = sp500.squeeze()
        return sp500

    def plot_results(self, portfolio_history: pd.DataFrame, filename='backtest_results.png'):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        # fetch S&P 500 benchmark
        start_date = portfolio_history.index[0]
        end_date = portfolio_history.index[-1]
        sp500 = self.fetch_sp500(start_date, end_date)

        initial_portfolio_value = portfolio_history['value'].iloc[0]
        portfolio_normalized = (portfolio_history['value'] / initial_portfolio_value) * 100

        sp500_aligned = sp500.reindex(portfolio_history.index, method='ffill')
        sp500_normalized = (sp500_aligned / sp500_aligned.iloc[0]) * 100

        ax1.plot(portfolio_history.index, portfolio_normalized, label='Momentum Strategy', linewidth=2)
        ax1.plot(portfolio_history.index, sp500_normalized, label='S&P 500', linewidth=2, alpha=0.7)
        ax1.set_title('Strategy vs S&P 500 (Normalized to 100)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value (Starting = 100)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(portfolio_history.index, portfolio_history['value'], color='green', linewidth=2)
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True, alpha=0.3)

        portfolio_returns = portfolio_history['value'].pct_change().dropna()
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        portfolio_running_max = portfolio_cumulative.expanding().max()
        portfolio_drawdown = (portfolio_cumulative / portfolio_running_max) - 1

        sp500_returns = sp500_aligned.pct_change().dropna()
        sp500_cumulative = (1 + sp500_returns).cumprod()
        sp500_running_max = sp500_cumulative.expanding().max()
        sp500_drawdown = (sp500_cumulative / sp500_running_max) - 1

        ax3.fill_between(portfolio_drawdown.index, portfolio_drawdown * 100, 0,
                         alpha=0.3, color='red', label='Momentum Strategy')
        ax3.plot(portfolio_drawdown.index, portfolio_drawdown * 100, color='red', linewidth=2)

        ax3.fill_between(sp500_drawdown.index, sp500_drawdown * 100, 0,
                         alpha=0.2, color='blue', label='S&P 500')
        ax3.plot(sp500_drawdown.index, sp500_drawdown * 100, color='blue',
                 linewidth=2, alpha=0.7, linestyle='--')

        ax3.set_title('Drawdown Comparison')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join('outputs', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.show()
