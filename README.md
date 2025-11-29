# Cryptocurrency Trading Algorithm

A complete machine learning-based trading system that predicts cryptocurrency price movements and generates automated buy/sell signals for Ethereum (ETH), Arbitrum (ARB), and other cryptocurrencies.

## Features

- **Data Collection**: Downloads OHLCV (Open, High, Low, Close, Volume) data from Binance, CoinGecko, or Bybit
- **Technical Indicators**: Calculates RSI, MACD, Moving Averages (SMA50, SMA200), Bollinger Bands, ATR, and more
- **Machine Learning Models**: Supports XGBoost, LightGBM, and LSTM neural networks
- **Trading Signals**: Generates buy/sell/hold signals based on model predictions
- **Backtesting**: Evaluates strategy performance with win rate, profit/loss, Sharpe ratio, and drawdown metrics
- **Visualization**: Plots price charts with signals, equity curves, and performance metrics
- **CLI Interface**: Easy-to-use command-line interface for all operations

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download Data
Download historical cryptocurrency data:
```bash
python main.py download --symbol ETH/USDT --timeframe 1d --days 365
```

For Arbitrum:
```bash
python main.py download --symbol ARB/USDT --timeframe 1d --days 365
```

### 2. Train a Model
Train an XGBoost model (recommended for speed):
```bash
python main.py train --symbol ETH/USDT --timeframe 1d --model xgboost
```

Train an LSTM model (slower but potentially more accurate):
```bash
python main.py train --symbol ETH/USDT --timeframe 1d --model lstm
```

### 3. Run Backtest
Evaluate the strategy on historical data:
```bash
python main.py backtest --symbol ETH/USDT --timeframe 1d --model xgboost --confidence 0.6
```

### 4. Make Predictions
Get real-time trading signals:
```bash
python main.py predict --symbol ETH/USDT --timeframe 1d --model xgboost --confidence 0.6
```

## Command Reference

### Download Command
```bash
python main.py download [OPTIONS]
```
Options:
- `--symbol`: Trading pair (e.g., ETH/USDT, ARB/USDT)
- `--timeframe`: Timeframe (1m, 5m, 1h, 1d, etc.)
- `--days`: Number of days of historical data
- `--exchange`: Exchange name (binance, bybit, coinbase)

### Train Command
```bash
python main.py train [OPTIONS]
```
Options:
- `--symbol`: Trading pair
- `--timeframe`: Timeframe
- `--model`: Model type (xgboost, lightgbm, lstm)
- `--days`: Number of days of training data

### Backtest Command
```bash
python main.py backtest [OPTIONS]
```
Options:
- `--symbol`: Trading pair
- `--timeframe`: Timeframe
- `--model`: Model type
- `--confidence`: Confidence threshold (0-1)
- `--days`: Number of days of data

### Predict Command
```bash
python main.py predict [OPTIONS]
```
Options:
- `--symbol`: Trading pair
- `--timeframe`: Timeframe
- `--model`: Model type
- `--confidence`: Confidence threshold (0-1)

## Project Structure

```
.
├── data_loader.py      # Downloads OHLCV data from crypto exchanges
├── features.py         # Generates technical indicators (RSI, MACD, SMA, etc.)
├── model.py            # ML model definition, training, and prediction
├── strategy.py         # Buy/sell signal generation logic
├── backtest.py         # Backtesting and performance evaluation
├── main.py             # CLI interface for running predictions
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Technical Indicators

The system calculates the following technical indicators:

- **RSI (Relative Strength Index)**: Momentum oscillator (14-period)
- **MACD**: Moving Average Convergence Divergence (12, 26, 9)
- **SMA**: Simple Moving Averages (50, 200 periods)
- **EMA**: Exponential Moving Averages (12, 26 periods)
- **Bollinger Bands**: Volatility bands with position and width
- **ATR**: Average True Range for volatility measurement
- **Price Changes**: Percentage changes over 1, 3, 7, 14 periods
- **Volume Features**: Volume ratios and moving averages

## Model Types

### XGBoost (Recommended)
- Fast training and prediction
- Good performance on tabular data
- Easy to interpret feature importance

### LightGBM
- Similar to XGBoost but often faster
- Good for large datasets

### LSTM
- Neural network for sequential data
- Slower but may capture complex patterns
- Requires more data and computational resources

## Backtesting Metrics

The backtester calculates:
- **Total Return**: Percentage return on initial capital
- **Win Rate**: Percentage of profitable trades
- **Profit/Loss**: Total profit or loss in USD
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Profit/Loss per Trade**: Mean profit per trade

## Trading Signals

- **BUY (1)**: Model predicts upward price movement with high confidence
- **SELL (-1)**: Model predicts downward price movement with high confidence
- **HOLD (0)**: Low confidence prediction or filtered by technical indicators

The strategy includes filters:
- **RSI Filter**: Avoids buying when RSI > 70 or selling when RSI < 30
- **Trend Filter**: Only buys in bullish trends (SMA50 > SMA200) and sells in bearish trends

## Example Workflow

1. **Download data for ETH and ARB**:
   ```bash
   python main.py download --symbol ETH/USDT --days 365
   python main.py download --symbol ARB/USDT --days 365
   ```

2. **Train models**:
   ```bash
   python main.py train --symbol ETH/USDT --model xgboost
   python main.py train --symbol ARB/USDT --model xgboost
   ```

3. **Backtest strategies**:
   ```bash
   python main.py backtest --symbol ETH/USDT --model xgboost
   python main.py backtest --symbol ARB/USDT --model xgboost
   ```

4. **Get real-time predictions**:
   ```bash
   python main.py predict --symbol ETH/USDT --model xgboost
   python main.py predict --symbol ARB/USDT --model xgboost
   ```

## Output Files

- `data/`: Downloaded OHLCV data (CSV files)
- `models/`: Trained model files (.pkl for XGBoost/LightGBM, .h5 for LSTM)
- `plots/`: Backtest visualization plots (PNG files)

## Notes

- **Internet Connection Required**: The system downloads data from cryptocurrency exchanges
- **Rate Limits**: The code respects exchange rate limits automatically
- **Data Quality**: Data is cleaned to remove invalid entries
- **Model Persistence**: Trained models are saved and can be reused
- **Confidence Threshold**: Adjust the confidence threshold to control signal frequency (higher = fewer but more confident signals)

## Disclaimer

This is an educational project for algorithmic trading research.
## License

This project is provided as-is for educational purposes.

