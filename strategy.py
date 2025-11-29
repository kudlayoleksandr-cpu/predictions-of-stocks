"""
Trading Strategy Module
Generates buy/sell signals based on model predictions and additional logic.
"""

import pandas as pd
import numpy as np


class TradingStrategy:
    """
    Class to generate trading signals based on ML model predictions.
    """
    
    def __init__(self, model, confidence_threshold=0.6):
        """
        Initialize the trading strategy.
        
        Args:
            model: Trained ML model (CryptoPredictor instance)
            confidence_threshold (float): Minimum probability threshold for signals (0-1)
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        print(f"Initialized trading strategy with confidence threshold: {confidence_threshold}")
    
    def generate_signals(self, df):
        """
        Generate buy/sell signals based on model predictions.
        
        Args:
            df (pd.DataFrame): DataFrame with features (must match training features)
        
        Returns:
            pd.DataFrame: DataFrame with added 'signal' column (1=buy, -1=sell, 0=hold)
        """
        # Make predictions
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)
        
        # Create signals DataFrame
        signals_df = df.copy()
        signals_df['prediction'] = predictions
        signals_df['probability'] = probabilities
        
        # Generate signals based on predictions and confidence
        signals = []
        for pred, prob in zip(predictions, probabilities):
            if prob >= self.confidence_threshold:
                # High confidence: buy if prediction is 1, sell if prediction is 0
                signal = 1 if pred == 1 else -1
            else:
                # Low confidence: hold
                signal = 0
            signals.append(signal)
        
        signals_df['signal'] = signals
        
        return signals_df
    
    def generate_signals_with_filters(self, df, use_rsi_filter=True, use_trend_filter=True):
        """
        Generate signals with additional technical filters.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            use_rsi_filter (bool): Filter out signals when RSI is extreme
            use_trend_filter (bool): Filter signals based on trend
        
        Returns:
            pd.DataFrame: DataFrame with filtered signals
        """
        # Get base signals
        signals_df = self.generate_signals(df)
        
        # RSI filter: avoid buying when RSI > 70, avoid selling when RSI < 30
        if use_rsi_filter and 'rsi' in signals_df.columns:
            rsi_filter = (
                ((signals_df['signal'] == 1) & (signals_df['rsi'] > 70)) |
                ((signals_df['signal'] == -1) & (signals_df['rsi'] < 30))
            )
            signals_df.loc[rsi_filter, 'signal'] = 0
        
        # Trend filter: use moving averages to confirm trend
        if use_trend_filter:
            if 'sma_50' in signals_df.columns and 'sma_200' in signals_df.columns:
                # Bullish trend: SMA50 > SMA200
                # Bearish trend: SMA50 < SMA200
                bullish_trend = signals_df['sma_50'] > signals_df['sma_200']
                
                # Filter: only buy in bullish trend, only sell in bearish trend
                trend_filter = (
                    ((signals_df['signal'] == 1) & ~bullish_trend) |
                    ((signals_df['signal'] == -1) & bullish_trend)
                )
                signals_df.loc[trend_filter, 'signal'] = 0
        
        return signals_df
    
    def get_signal_summary(self, signals_df):
        """
        Get summary statistics of generated signals.
        
        Args:
            signals_df (pd.DataFrame): DataFrame with signals
        
        Returns:
            dict: Summary statistics
        """
        signal_counts = signals_df['signal'].value_counts()
        
        summary = {
            'total_signals': len(signals_df),
            'buy_signals': signal_counts.get(1, 0),
            'sell_signals': signal_counts.get(-1, 0),
            'hold_signals': signal_counts.get(0, 0),
            'buy_percentage': (signal_counts.get(1, 0) / len(signals_df)) * 100,
            'sell_percentage': (signal_counts.get(-1, 0) / len(signals_df)) * 100,
            'hold_percentage': (signal_counts.get(0, 0) / len(signals_df)) * 100,
        }
        
        return summary
    
    def print_signal_summary(self, signals_df):
        """
        Print a formatted summary of signals.
        
        Args:
            signals_df (pd.DataFrame): DataFrame with signals
        """
        summary = self.get_signal_summary(signals_df)
        
        print("\n" + "="*50)
        print("TRADING SIGNAL SUMMARY")
        print("="*50)
        print(f"Total periods: {summary['total_signals']}")
        print(f"Buy signals: {summary['buy_signals']} ({summary['buy_percentage']:.2f}%)")
        print(f"Sell signals: {summary['sell_signals']} ({summary['sell_percentage']:.2f}%)")
        print(f"Hold signals: {summary['hold_signals']} ({summary['hold_percentage']:.2f}%)")
        print("="*50 + "\n")


if __name__ == "__main__":
    # Example usage
    from data_loader import CryptoDataLoader
    from features import FeatureEngineer
    from model import CryptoPredictor
    
    # Load data
    loader = CryptoDataLoader('binance')
    df = loader.get_ohlcv_data('ETH/USDT', timeframe='1d', days=365)
    df = loader.clean_data(df)
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    
    # Load trained model (assuming it exists)
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
    
    # Print summary
    strategy.print_signal_summary(signals_df)
    
    # Show recent signals
    print("\nRecent signals:")
    print(signals_df[['close', 'prediction', 'probability', 'signal']].tail(20))

