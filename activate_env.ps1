# PowerShell script to activate the virtual environment on D drive
& D:\vgtu_env\Scripts\Activate.ps1
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host "All packages are installed on D drive (D:\vgtu_env)" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run:" -ForegroundColor Yellow
Write-Host "  python main.py download --symbol ETH/USDT"
Write-Host "  python main.py train --symbol ETH/USDT --model xgboost"
Write-Host "  python main.py backtest --symbol ETH/USDT --model xgboost"
Write-Host "  python main.py predict --symbol ETH/USDT --model xgboost"

