"""
Example Usage Script
Demonstrates how to use the trading algorithm components.
"""

from data_loader import CryptoDataLoader
from features import FeatureEngineer
from model import CryptoPredictor
from strategy import TradingStrategy
from backtest import Backtester


def example_workflow():
    """
    Complete example workflow: download data, create features, train model, 
    generate signals, and backtest.
    """
    print("="*60)
    print("CRYPTOCURRENCY TRADING ALGORITHM - EXAMPLE WORKFLOW")
    print("="*60)
    
    # Step 1: Download data
    print("\n[Step 1] Downloading data...")
    loader = CryptoDataLoader('binance')
    df = loader.get_ohlcv_data('ETH/USDT', timeframe='1d', days=365)
    df = loader.clean_data(df)
    print(f"Downloaded {len(df)} candles")
    
    # Step 2: Create features
    print("\n[Step 2] Creating technical indicators...")
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df)
    features_df['target'] = engineer.create_target(features_df, prediction_horizon=1)
    print(f"Created {len(features_df.columns)} features")
    
    # Step 3: Train model
    print("\n[Step 3] Training XGBoost model...")
    predictor = CryptoPredictor(model_type='xgboost')
    accuracy = predictor.train(features_df)
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    # Step 4: Generate signals
    print("\n[Step 4] Generating trading signals...")
    strategy = TradingStrategy(predictor, confidence_threshold=0.6)
    signals_df = strategy.generate_signals_with_filters(features_df)
    strategy.print_signal_summary(signals_df)
    
    # Step 5: Run backtest
    print("\n[Step 5] Running backtest...")
    backtester = Backtester(initial_capital=10000, commission=0.001)
    results_df = backtester.run_backtest(signals_df)
    backtester.print_metrics(results_df)
    
    # Step 6: Visualize results
    print("\n[Step 6] Generating visualization...")
    backtester.plot_results(results_df, save_path='example_backtest_results.png')
    
    print("\n" + "="*60)
    print("Example workflow completed!")
    print("="*60)


if __name__ == "__main__":
    example_workflow()

