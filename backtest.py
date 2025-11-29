"""
Backtesting Module
Evaluates trading strategy performance on historical data.
Calculates win rate, profit/loss, and other performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Backtester:
    """
    Class to backtest trading strategies on historical data.
    """
    
    def __init__(self, initial_capital=10000, commission=0.001):
        """
        Initialize the backtester.
        
        Args:
            initial_capital (float): Starting capital in USD
            commission (float): Trading commission rate (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        print(f"Initialized backtester with ${initial_capital:,.2f} capital and {commission*100:.2f}% commission")
    
    def run_backtest(self, signals_df, price_col='close'):
        """
        Run backtest on historical data with trading signals.
        
        Args:
            signals_df (pd.DataFrame): DataFrame with 'signal' column and price data
            price_col (str): Name of price column to use
        
        Returns:
            pd.DataFrame: DataFrame with backtest results
        """
        # Create a copy to avoid modifying original
        results_df = signals_df.copy()
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Number of coins held
        entry_price = 0
        trades = []
        equity_curve = [capital]
        
        # Track state
        in_position = False
        
        print("Running backtest...")
        
        for i, row in results_df.iterrows():
            signal = row['signal']
            price = row[price_col]
            
            # Execute trades based on signals
            if signal == 1 and not in_position:  # Buy signal
                # Calculate how many coins we can buy
                position = (capital * (1 - self.commission)) / price
                entry_price = price
                capital = 0
                in_position = True
                trades.append({
                    'timestamp': i,
                    'type': 'BUY',
                    'price': price,
                    'position': position,
                    'capital': capital
                })
            
            elif signal == -1 and in_position:  # Sell signal
                # Sell all coins
                capital = (position * price) * (1 - self.commission)
                profit = capital - (position * entry_price)
                position = 0
                in_position = False
                trades.append({
                    'timestamp': i,
                    'type': 'SELL',
                    'price': price,
                    'profit': profit,
                    'capital': capital
                })
            
            # Calculate current equity (capital + position value)
            if in_position:
                current_equity = position * price
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        # Close any open position at the end
        if in_position:
            final_price = results_df[price_col].iloc[-1]
            capital = (position * final_price) * (1 - self.commission)
            profit = capital - (position * entry_price)
            trades.append({
                'timestamp': results_df.index[-1],
                'type': 'SELL',
                'price': final_price,
                'profit': profit,
                'capital': capital
            })
            equity_curve[-1] = capital
        
        # Add equity curve to results
        results_df['equity'] = equity_curve[1:]  # Skip first value (initial capital)
        results_df['returns'] = results_df['equity'].pct_change()
        results_df['cumulative_returns'] = (results_df['equity'] / self.initial_capital - 1) * 100
        
        # Store trades
        self.trades = pd.DataFrame(trades)
        
        print(f"Backtest completed. {len(self.trades)} trades executed.")
        
        return results_df
    
    def calculate_metrics(self, results_df):
        """
        Calculate performance metrics from backtest results.
        
        Args:
            results_df (pd.DataFrame): DataFrame with backtest results
        
        Returns:
            dict: Dictionary of performance metrics
        """
        final_equity = results_df['equity'].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        # Calculate win rate from trades
        if len(self.trades) > 0:
            # Pair buy and sell trades
            buy_trades = self.trades[self.trades['type'] == 'BUY']
            sell_trades = self.trades[self.trades['type'] == 'SELL']
            
            if len(sell_trades) > 0:
                profits = sell_trades['profit'].values
                winning_trades = len(profits[profits > 0])
                total_trades = len(profits)
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                avg_profit = profits.mean()
                avg_win = profits[profits > 0].mean() if len(profits[profits > 0]) > 0 else 0
                avg_loss = profits[profits < 0].mean() if len(profits[profits < 0]) > 0 else 0
                max_profit = profits.max()
                max_loss = profits.min()
            else:
                win_rate = 0
                avg_profit = 0
                avg_win = 0
                avg_loss = 0
                max_profit = 0
                max_loss = 0
        else:
            win_rate = 0
            avg_profit = 0
            avg_win = 0
            avg_loss = 0
            max_profit = 0
            max_loss = 0
        
        # Calculate Sharpe ratio (simplified)
        returns = results_df['returns'].dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        equity = results_df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'total_profit_loss': final_equity - self.initial_capital,
            'win_rate': win_rate,
            'total_trades': len(sell_trades) if len(self.trades) > 0 else 0,
            'winning_trades': winning_trades if len(self.trades) > 0 else 0,
            'losing_trades': total_trades - winning_trades if len(self.trades) > 0 else 0,
            'avg_profit_per_trade': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown
        }
        
        return metrics
    
    def print_metrics(self, results_df):
        """
        Print formatted performance metrics.
        
        Args:
            results_df (pd.DataFrame): DataFrame with backtest results
        """
        metrics = self.calculate_metrics(results_df)
        
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE METRICS")
        print("="*60)
        print(f"Initial Capital:        ${metrics['initial_capital']:,.2f}")
        print(f"Final Equity:           ${metrics['final_equity']:,.2f}")
        print(f"Total Return:           {metrics['total_return_pct']:.2f}%")
        print(f"Profit/Loss:            ${metrics['total_profit_loss']:,.2f}")
        print(f"\nTrade Statistics:")
        print(f"  Total Trades:         {metrics['total_trades']}")
        print(f"  Winning Trades:       {metrics['winning_trades']}")
        print(f"  Losing Trades:        {metrics['losing_trades']}")
        print(f"  Win Rate:             {metrics['win_rate']:.2f}%")
        print(f"\nProfit/Loss per Trade:")
        print(f"  Average:              ${metrics['avg_profit_per_trade']:,.2f}")
        print(f"  Average Win:          ${metrics['avg_win']:,.2f}")
        print(f"  Average Loss:         ${metrics['avg_loss']:,.2f}")
        print(f"  Max Profit:           ${metrics['max_profit']:,.2f}")
        print(f"  Max Loss:             ${metrics['max_loss']:,.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:         {metrics['max_drawdown_pct']:.2f}%")
        print("="*60 + "\n")
    
    def plot_results(self, results_df, price_col='close', save_path=None):
        """
        Plot backtest results including equity curve and price with signals.
        
        Args:
            results_df (pd.DataFrame): DataFrame with backtest results
            price_col (str): Name of price column
            save_path (str): Optional path to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Price with buy/sell signals
        axes[0].plot(results_df.index, results_df[price_col], label='Price', linewidth=1.5, color='black')
        
        buy_signals = results_df[results_df['signal'] == 1]
        sell_signals = results_df[results_df['signal'] == -1]
        
        if len(buy_signals) > 0:
            axes[0].scatter(buy_signals.index, buy_signals[price_col], 
                          color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        if len(sell_signals) > 0:
            axes[0].scatter(sell_signals.index, sell_signals[price_col], 
                          color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        axes[0].set_title('Price Chart with Trading Signals', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price (USD)', fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Equity curve
        axes[1].plot(results_df.index, results_df['equity'], 
                    label='Equity Curve', linewidth=2, color='blue')
        axes[1].axhline(y=self.initial_capital, color='gray', linestyle='--', 
                       label='Initial Capital', alpha=0.7)
        axes[1].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Equity (USD)', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative returns
        axes[2].plot(results_df.index, results_df['cumulative_returns'], 
                    label='Cumulative Returns', linewidth=2, color='purple')
        axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        axes[2].set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].set_ylabel('Returns (%)', fontsize=12)
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    from data_loader import CryptoDataLoader
    from features import FeatureEngineer
    from model import CryptoPredictor
    from strategy import TradingStrategy
    
    # Load data
    loader = CryptoDataLoader('binance')
    df = loader.get_ohlcv_data('ETH/USDT', timeframe='1d', days=365)
    df = loader.clean_data(df)
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    
    # Load or train model
    predictor = CryptoPredictor(model_type='xgboost')
    try:
        predictor.load_model('models/eth_xgboost.pkl')
    except:
        print("Model not found. Training new model...")
        features_df['target'] = engineer.create_target(features_df, prediction_horizon=1)
        predictor.train(features_df)
    
    # Generate signals
    strategy = TradingStrategy(predictor, confidence_threshold=0.6)
    signals_df = strategy.generate_signals_with_filters(features_df)
    
    # Run backtest
    backtester = Backtester(initial_capital=10000, commission=0.001)
    results_df = backtester.run_backtest(signals_df)
    
    # Print metrics
    backtester.print_metrics(results_df)
    
    # Plot results
    backtester.plot_results(results_df, save_path='backtest_results.png')

