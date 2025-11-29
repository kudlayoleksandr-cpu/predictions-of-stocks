"""
Main CLI Module
Command-line interface for running cryptocurrency trading predictions.
"""

import argparse
import os
import sys
from datetime import datetime
import pandas as pd

from data_loader import CryptoDataLoader
from features import FeatureEngineer
from model import CryptoPredictor
from strategy import TradingStrategy
from backtest import Backtester


def download_data(symbol, timeframe, days, exchange='binance'):
    """
    Download and save cryptocurrency data.
    
    Args:
        symbol (str): Trading pair (e.g., 'ETH/USDT')
        timeframe (str): Timeframe ('1d', '1h', etc.)
        days (int): Number of days of data
        exchange (str): Exchange name
    """
    print(f"\n{'='*60}")
    print(f"Downloading {symbol} data from {exchange}")
    print(f"{'='*60}\n")
    
    loader = CryptoDataLoader(exchange)
    df = loader.get_ohlcv_data(symbol, timeframe=timeframe, days=days)
    df = loader.clean_data(df)
    
    # Save data
    filename = f"data/{symbol.replace('/', '_')}_{timeframe}.csv"
    os.makedirs('data', exist_ok=True)
    loader.save_data(df, filename)
    
    return df


def train_model(symbol, timeframe, model_type, days=365):
    """
    Train a machine learning model on historical data.
    
    Args:
        symbol (str): Trading pair
        timeframe (str): Timeframe
        model_type (str): Model type ('xgboost', 'lightgbm', 'lstm')
        days (int): Number of days of training data
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model for {symbol}")
    print(f"{'='*60}\n")
    
    # Load data
    loader = CryptoDataLoader('binance')
    df = loader.get_ohlcv_data(symbol, timeframe=timeframe, days=days)
    df = loader.clean_data(df)
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    features_df['target'] = engineer.create_target(features_df, prediction_horizon=1)
    
    # Train model
    predictor = CryptoPredictor(model_type=model_type)
    accuracy = predictor.train(features_df)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f"models/{symbol.replace('/', '_')}_{timeframe}_{model_type}.pkl"
    if model_type == 'lstm':
        model_path = model_path.replace('.pkl', '.h5')
    predictor.save_model(model_path)
    
    print(f"\nModel saved to {model_path}")
    print(f"Model accuracy: {accuracy:.4f}")
    
    return predictor


def run_backtest(symbol, timeframe, model_type, confidence_threshold=0.6, days=365):
    """
    Run backtest on historical data.
    
    Args:
        symbol (str): Trading pair
        timeframe (str): Timeframe
        model_type (str): Model type
        confidence_threshold (float): Confidence threshold for signals
        days (int): Number of days of data
    """
    print(f"\n{'='*60}")
    print(f"Running backtest for {symbol}")
    print(f"{'='*60}\n")
    
    # Load data
    loader = CryptoDataLoader('binance')
    df = loader.get_ohlcv_data(symbol, timeframe=timeframe, days=days)
    df = loader.clean_data(df)
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    
    # Load model
    model_path = f"models/{symbol.replace('/', '_')}_{timeframe}_{model_type}.pkl"
    if model_type == 'lstm':
        model_path = model_path.replace('.pkl', '.h5')
    
    predictor = CryptoPredictor(model_type=model_type)
    try:
        predictor.load_model(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Training new model...")
        features_df['target'] = engineer.create_target(features_df, prediction_horizon=1)
        predictor.train(features_df)
        predictor.save_model(model_path)
    
    # Generate signals
    strategy = TradingStrategy(predictor, confidence_threshold=confidence_threshold)
    signals_df = strategy.generate_signals_with_filters(features_df)
    strategy.print_signal_summary(signals_df)
    
    # Run backtest
    backtester = Backtester(initial_capital=10000, commission=0.001)
    results_df = backtester.run_backtest(signals_df)
    
    # Print metrics
    backtester.print_metrics(results_df)
    
    # Plot results
    os.makedirs('plots', exist_ok=True)
    plot_path = f"plots/backtest_{symbol.replace('/', '_')}_{timeframe}.png"
    backtester.plot_results(results_df, save_path=plot_path)
    
    return results_df


def predict_realtime(symbol, timeframe, model_type, confidence_threshold=0.6):
    """
    Make real-time predictions for a cryptocurrency.
    
    Args:
        symbol (str): Trading pair
        timeframe (str): Timeframe
        model_type (str): Model type
        confidence_threshold (float): Confidence threshold
    """
    print(f"\n{'='*60}")
    print(f"Real-time Prediction for {symbol}")
    print(f"{'='*60}\n")
    
    # Load recent data (last 365 days to calculate indicators)
    loader = CryptoDataLoader('binance')
    df = loader.get_ohlcv_data(symbol, timeframe=timeframe, days=365)
    df = loader.clean_data(df)
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    
    # Load model
    model_path = f"models/{symbol.replace('/', '_')}_{timeframe}_{model_type}.pkl"
    if model_type == 'lstm':
        model_path = model_path.replace('.pkl', '.h5')
    
    predictor = CryptoPredictor(model_type=model_type)
    try:
        predictor.load_model(model_path)
        print(f"Loaded model from {model_path}\n")
    except FileNotFoundError:
        print(f"Model not found. Please train a model first using: python main.py train")
        return
    
    # Get latest data point for prediction
    latest_features = features_df.iloc[-1:].copy()
    
    # Make prediction
    prediction = predictor.predict(latest_features)[0]
    probability = predictor.predict_proba(latest_features)[0]
    
    # Generate signal
    strategy = TradingStrategy(predictor, confidence_threshold=confidence_threshold)
    signals_df = strategy.generate_signals_with_filters(latest_features)
    signal = signals_df['signal'].iloc[0]
    
    # Display results
    current_price = latest_features['close'].iloc[0]
    current_time = latest_features.index[0]
    
    print(f"Timestamp:        {current_time}")
    print(f"Current Price:     ${current_price:.2f}")
    print(f"Prediction:        {'UP' if prediction == 1 else 'DOWN'}")
    print(f"Confidence:        {probability:.2%}")
    print(f"Signal:            ", end='')
    
    if signal == 1:
        print("🟢 BUY")
    elif signal == -1:
        print("🔴 SELL")
    else:
        print("⚪ HOLD (low confidence)")
    
    # Show recent indicators
    if 'rsi' in latest_features.columns:
        print(f"RSI:               {latest_features['rsi'].iloc[0]:.2f}")
    if 'sma_50' in latest_features.columns and 'sma_200' in latest_features.columns:
        sma50 = latest_features['sma_50'].iloc[0]
        sma200 = latest_features['sma_200'].iloc[0]
        trend = "BULLISH" if sma50 > sma200 else "BEARISH"
        print(f"Trend (SMA50/200): {trend}")
    
    print(f"\n{'='*60}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Cryptocurrency Trading Algorithm - ML-based price prediction and trading signals'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download cryptocurrency data')
    download_parser.add_argument('--symbol', type=str, default='ETH/USDT', 
                                help='Trading pair (default: ETH/USDT)')
    download_parser.add_argument('--timeframe', type=str, default='1d', 
                                help='Timeframe: 1m, 5m, 1h, 1d, etc. (default: 1d)')
    download_parser.add_argument('--days', type=int, default=365, 
                                help='Number of days of data (default: 365)')
    download_parser.add_argument('--exchange', type=str, default='binance', 
                                help='Exchange name (default: binance)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML model')
    train_parser.add_argument('--symbol', type=str, default='ETH/USDT', 
                             help='Trading pair (default: ETH/USDT)')
    train_parser.add_argument('--timeframe', type=str, default='1d', 
                             help='Timeframe (default: 1d)')
    train_parser.add_argument('--model', type=str, default='xgboost', 
                             choices=['xgboost', 'lightgbm', 'lstm'],
                             help='Model type (default: xgboost)')
    train_parser.add_argument('--days', type=int, default=365, 
                             help='Number of days of training data (default: 365)')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--symbol', type=str, default='ETH/USDT', 
                                help='Trading pair (default: ETH/USDT)')
    backtest_parser.add_argument('--timeframe', type=str, default='1d', 
                                help='Timeframe (default: 1d)')
    backtest_parser.add_argument('--model', type=str, default='xgboost', 
                                choices=['xgboost', 'lightgbm', 'lstm'],
                                help='Model type (default: xgboost)')
    backtest_parser.add_argument('--confidence', type=float, default=0.6, 
                                help='Confidence threshold (default: 0.6)')
    backtest_parser.add_argument('--days', type=int, default=365, 
                                help='Number of days of data (default: 365)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make real-time prediction')
    predict_parser.add_argument('--symbol', type=str, default='ETH/USDT', 
                               help='Trading pair (default: ETH/USDT)')
    predict_parser.add_argument('--timeframe', type=str, default='1d', 
                               help='Timeframe (default: 1d)')
    predict_parser.add_argument('--model', type=str, default='xgboost', 
                               choices=['xgboost', 'lightgbm', 'lstm'],
                               help='Model type (default: xgboost)')
    predict_parser.add_argument('--confidence', type=float, default=0.6, 
                               help='Confidence threshold (default: 0.6)')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        download_data(args.symbol, args.timeframe, args.days, args.exchange)
    
    elif args.command == 'train':
        train_model(args.symbol, args.timeframe, args.model, args.days)
    
    elif args.command == 'backtest':
        run_backtest(args.symbol, args.timeframe, args.model, args.confidence, args.days)
    
    elif args.command == 'predict':
        predict_realtime(args.symbol, args.timeframe, args.model, args.confidence)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

