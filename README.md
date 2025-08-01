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

## The strategy:
- Downloads SPY, TLT, and GLD data from Yahoo Finance (2018–2024).
- Computes static and dynamic risk parity portfolios.
- Uses macro indicators (VIX, Yield Spread) and a trained Random Forest classifier to predict market turbulence.
- Adjusts leverage dynamically to manage risk.
- Includes transaction cost modeling and COVID stress testing.

## Machine Learning Component
- Model: `RandomForestClassifier`
- Features:
  - VIX
  - Yield Spread
  - 6-month Rolling Sharpe
  - Portfolio Turnover
- Target:
  - Labeled turbulence = 5-day forward return < -2%

## How It Works
- **Risk Parity Weights**: Computed using inverse volatility.
- **Volatility Targeting**: Portfolio leverage scaled to maintain 10% annualized volatility.
- **Macro Filters**:
  - VIX (volatility index) dampens exposure when above 20.
  - Yield Spread (10y - 2y) dampens exposure when inverted (≤ 0).
- **Machine Learning Layer**:
  - Random Forest Classifier trained to predict turbulence (forward 5-day returns < -2%).
  - Input Features: VIX, Yield Spread, Rolling Sharpe, Turnover.
  - If turbulence probability > 0.6 → reduce leverage; > 0.8 → reduce further.
- **Transaction Costs**: Included based on turnover.
  
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


## Run the strategy:
Clone this repo and run:

```bash
python ml_risk_parity.py

