@echo off
REM Activate the virtual environment on D drive
call D:\vgtu_env\Scripts\activate.bat
echo Virtual environment activated!
echo All packages are installed on D drive (D:\vgtu_env)
echo.
echo You can now run:
echo   python main.py download --symbol ETH/USDT
echo   python main.py train --symbol ETH/USDT --model xgboost
echo   python main.py backtest --symbol ETH/USDT --model xgboost
echo   python main.py predict --symbol ETH/USDT --model xgboost

