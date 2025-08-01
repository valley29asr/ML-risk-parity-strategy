Author : Anshruta Singh Rahar
Undergraduate, Penn State University
B.S in Computer Science and Computational Maths, Minor in Statistics

## ML-risk-parity-strategy
This project implements a **Machine Learning–Enhanced Risk Parity Strategy** using historical market data from SPY (Equities), TLT (Treasuries), and GLD (Gold).

## Summary
- Allocates capital using **risk parity** based on rolling volatilities.
- Dynamically adjusts **portfolio leverage** using:
  - VIX (market volatility index)
  - Yield Curve Spread (10Y - 2Y)
  - **Random Forest Classifier** to detect upcoming market turbulence.
- Applies leverage dampening when turbulence probability is high.
- Incorporates **transaction costs**, **turnover analysis**, and **risk contributions**.
- Benchmarked against **Static RP** and **60/40 Portfolio**.

 
## Strategies Compared
- **Static Risk Parity**  
- **Dynamic RP with Macro Filters (VIX & Yield Spread)**  
- **Dynamic RP with ML Turbulence Prediction**  
- **Traditional 60/40 Portfolio**

## Machine Learning Component
- Model: `RandomForestClassifier`
- Features:
  - VIX
  - Yield Spread
  - 6-month Rolling Sharpe
  - Portfolio Turnover
- Target:
  - Labeled turbulence = 5-day forward return < -2%


## Performance Outputs
- Cumulative return comparison plot
- Sharpe ratio & max drawdown metrics
- Confusion matrix & ROC curve
- Feature importance plot

## Files
- `ml_risk_parity.py`: Full strategy code
- `ml_strategy_returns.csv`: Exported returns from ML-enhanced strategy

## Requirements
```bash
pip install yfinance pandas matplotlib seaborn scikit-learn pandas_datareader
