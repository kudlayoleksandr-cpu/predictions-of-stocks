"""
Data Loader Module
Downloads OHLCV (Open, High, Low, Close, Volume) data from cryptocurrency exchanges.
Supports multiple exchanges via CCXT library.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time


class CryptoDataLoader:
    """
    Class to download cryptocurrency market data from various exchanges.
    """
    
    def __init__(self, exchange_name='binance'):
        """
        Initialize the data loader with an exchange.
        
        Args:
            exchange_name (str): Name of the exchange ('binance', 'bybit', 'coinbase', etc.)
        """
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class({
            'enableRateLimit': True,  # Respect rate limits
        })
        
        print(f"Initialized {exchange_name} exchange connection")
    
    def get_ohlcv_data(self, symbol, timeframe='1d', days=365):
        """
        Download OHLCV data for a given cryptocurrency symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'ETH/USDT', 'ARB/USDT')
            timeframe (str): Timeframe for candles ('1m', '5m', '1h', '1d', etc.)
            days (int): Number of days of historical data to fetch
        
        Returns:
            pd.DataFrame: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        try:
            # Calculate since timestamp (days ago)
            since = self.exchange.milliseconds() - days * 24 * 60 * 60 * 1000
            
            print(f"Downloading {symbol} data for last {days} days...")
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure data is sorted by timestamp
            df.sort_index(inplace=True)
            
            print(f"Downloaded {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return pd.DataFrame()
    
    def clean_data(self, df):
        """
        Clean and preprocess the OHLCV data.
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        if df.empty:
            return df
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove rows with zero volume (likely bad data)
        df = df[df['volume'] > 0]
        
        # Remove rows with invalid prices
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        
        # Ensure high >= low
        df = df[df['high'] >= df['low']]
        
        # Forward fill missing values
        df.ffill(inplace=True)
        
        # Drop any remaining NaN values
        df.dropna(inplace=True)
        
        print(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    def save_data(self, df, filename):
        """
        Save DataFrame to CSV file.
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str): Output filename
        """
        df.to_csv(filename)
        print(f"Data saved to {filename}")
    
    def load_data(self, filename):
        """
        Load DataFrame from CSV file.
        
        Args:
            filename (str): Input filename
        
        Returns:
            pd.DataFrame: Loaded data
        """
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Data loaded from {filename}: {len(df)} rows")
        return df


if __name__ == "__main__":
    # Example usage
    loader = CryptoDataLoader('binance')
    
    # Download ETH data
    eth_data = loader.get_ohlcv_data('ETH/USDT', timeframe='1d', days=365)
    eth_data = loader.clean_data(eth_data)
    loader.save_data(eth_data, 'eth_data.csv')
    
    # Download ARB data
    arb_data = loader.get_ohlcv_data('ARB/USDT', timeframe='1d', days=365)
    arb_data = loader.clean_data(arb_data)
    loader.save_data(arb_data, 'arb_data.csv')

