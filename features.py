"""
Feature Engineering Module
Generates technical indicators from OHLCV data for machine learning models.
Includes RSI, MACD, Moving Averages, and other common indicators.
"""

import pandas as pd
import numpy as np
import ta


class FeatureEngineer:
    """
    Class to generate technical indicators and features from OHLCV data.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        pass
    
    def calculate_rsi(self, df, period=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            df (pd.DataFrame): OHLCV data with 'close' column
            period (int): RSI period (default 14)
        
        Returns:
            pd.Series: RSI values
        """
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=period)
        return rsi.rsi()
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df (pd.DataFrame): OHLCV data with 'close' column
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line period
        
        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        macd = ta.trend.MACD(close=df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        return macd.macd(), macd.macd_signal(), macd.macd_diff()
    
    def calculate_sma(self, df, period):
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            df (pd.DataFrame): OHLCV data with 'close' column
            period (int): SMA period
        
        Returns:
            pd.Series: SMA values
        """
        return ta.trend.SMAIndicator(close=df['close'], window=period).sma_indicator()
    
    def calculate_ema(self, df, period):
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            df (pd.DataFrame): OHLCV data with 'close' column
            period (int): EMA period
        
        Returns:
            pd.Series: EMA values
        """
        return ta.trend.EMAIndicator(close=df['close'], window=period).ema_indicator()
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """
        Calculate Bollinger Bands.
        
        Args:
            df (pd.DataFrame): OHLCV data with 'close' column
            period (int): Moving average period
            std_dev (float): Standard deviation multiplier
        
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        bb = ta.volatility.BollingerBands(close=df['close'], window=period, window_dev=std_dev)
        return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
    
    def calculate_atr(self, df, period=14):
        """
        Calculate Average True Range (ATR) for volatility.
        
        Args:
            df (pd.DataFrame): OHLCV data
            period (int): ATR period
        
        Returns:
            pd.Series: ATR values
        """
        atr = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=period
        )
        return atr.average_true_range()
    
    def calculate_price_change(self, df, periods=[1, 3, 7]):
        """
        Calculate price change percentage over different periods.
        
        Args:
            df (pd.DataFrame): OHLCV data with 'close' column
            periods (list): List of periods to calculate change for
        
        Returns:
            dict: Dictionary of price change series
        """
        changes = {}
        for period in periods:
            changes[f'price_change_{period}'] = df['close'].pct_change(period) * 100
        return changes
    
    def calculate_volume_features(self, df, periods=[10, 20, 50]):
        """
        Calculate volume-based features.
        
        Args:
            df (pd.DataFrame): OHLCV data with 'volume' column
            periods (list): List of periods for volume moving averages
        
        Returns:
            dict: Dictionary of volume feature series
        """
        features = {}
        for period in periods:
            features[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / features[f'volume_sma_{period}']
        return features
    
    def create_all_features(self, df):
        """
        Create all technical indicators and features from OHLCV data.
        
        Args:
            df (pd.DataFrame): OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            pd.DataFrame: DataFrame with original data and all features
        """
        # Create a copy to avoid modifying original
        features_df = df.copy()
        
        print("Calculating technical indicators...")
        
        # RSI
        features_df['rsi'] = self.calculate_rsi(features_df, period=14)
        
        # MACD
        macd, macd_signal, macd_hist = self.calculate_macd(features_df)
        features_df['macd'] = macd
        features_df['macd_signal'] = macd_signal
        features_df['macd_histogram'] = macd_hist
        
        # Moving Averages
        features_df['sma_50'] = self.calculate_sma(features_df, period=50)
        features_df['sma_200'] = self.calculate_sma(features_df, period=200)
        features_df['ema_12'] = self.calculate_ema(features_df, period=12)
        features_df['ema_26'] = self.calculate_ema(features_df, period=26)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(features_df)
        features_df['bb_upper'] = bb_upper
        features_df['bb_middle'] = bb_middle
        features_df['bb_lower'] = bb_lower
        features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features_df['bb_position'] = (features_df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        features_df['atr'] = self.calculate_atr(features_df, period=14)
        features_df['atr_percent'] = (features_df['atr'] / features_df['close']) * 100
        
        # Price changes
        price_changes = self.calculate_price_change(features_df, periods=[1, 3, 7, 14])
        for key, value in price_changes.items():
            features_df[key] = value
        
        # Volume features
        volume_features = self.calculate_volume_features(features_df, periods=[10, 20, 50])
        for key, value in volume_features.items():
            features_df[key] = value
        
        # Additional features
        features_df['high_low_ratio'] = features_df['high'] / features_df['low']
        features_df['close_open_ratio'] = features_df['close'] / features_df['open']
        
        # Drop rows with NaN values (from indicator calculations)
        features_df.dropna(inplace=True)
        
        print(f"Created {len(features_df.columns)} features. Final shape: {features_df.shape}")
        
        return features_df
    
    def create_target(self, df, prediction_horizon=1):
        """
        Create target variable for prediction (future price movement).
        
        Args:
            df (pd.DataFrame): OHLCV data with 'close' column
            prediction_horizon (int): Number of periods ahead to predict
        
        Returns:
            pd.Series: Target values (1 for upward movement, 0 for downward)
        """
        # Calculate future price change
        future_price = df['close'].shift(-prediction_horizon)
        price_change = (future_price - df['close']) / df['close'] * 100
        
        # Create binary target: 1 if price goes up, 0 if price goes down
        target = (price_change > 0).astype(int)
        
        return target


if __name__ == "__main__":
    # Example usage
    from data_loader import CryptoDataLoader
    
    # Load data
    loader = CryptoDataLoader('binance')
    df = loader.get_ohlcv_data('ETH/USDT', timeframe='1d', days=365)
    df = loader.clean_data(df)
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    
    # Create target
    target = engineer.create_target(features_df, prediction_horizon=1)
    features_df['target'] = target
    
    print(features_df.head())
    print(f"\nFeature columns: {list(features_df.columns)}")

