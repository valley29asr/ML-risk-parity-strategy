"""
ML-RP: Machine Learning–Enhanced Risk Parity Strategy
Author: Anshruta Singh Rahar

Description: Builds a dynamic, risk-managed portfolio across SPY, TLT, and GLD using risk parity allocation, 
macro regime filters (VIX & yield spread), and machine learning to predict turbulence.
Leverage is adjusted based on model-predicted probabilities to optimize returns and control drawdowns.
"""


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
import seaborn as sns

#dates/year that the model is working with for data
start_date = datetime.datetime(2018,1,1)
end_date = datetime.datetime(2024,12,31)

#10Y and 2Y treasury yields used to compute yield curve spread (macro regime filter)
yield_10y = pdr.DataReader('GS10', 'fred', start_date, end_date)
yield_2y = pdr.DataReader('GS2', 'fred', start_date, end_date)

#downloading asset prices from yahoo finance
asset = ['SPY', 'GLD', 'TLT']
data = yf.download(asset, start='2018-01-01', end = '2024-12-31', auto_adjust=False)
adj_close = data['Adj Close'].copy().dropna()

#graph to visualize the prices of the assets throughout the years
adj_close.plot(figsize=(12, 6), title='Price History of SPY, TLT, and GLD')
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price (USD)")
plt.grid(True)
plt.show()


#the model's target volatilty
target_annual_vol = 0.10
trading_days  = 252

#computing returns
returns = adj_close.pct_change().dropna()


# getting VIX & yield spread as macro regime indicators
# - high VIX → elevated market fear → reduce leverage
# - yield curve inversion (spread <= 0) → recession signal → reduce leverage
vix_data = yf.download('^VIX', start = '2018-01-01', end = '2024-12-31')
vix_level = vix_data['Close'].reindex(returns.index).fillna(method='ffill')
vix_level = vix_level.squeeze()

yield_spread = (yield_10y['GS10'] - yield_2y['GS2']).rename('Yield_Spread')
yield_spread = yield_spread.reindex(returns.index).fillna(method='ffill')

'dynamic risk parity'
#computing risk-parity weights based on inverse volatility 
def compute_risk_parity_wt(returns_wnd):
    vol = returns_wnd.std()
    inv_vol  = 1/vol
    wts = inv_vol/inv_vol.sum()
    return wts


'traditional 60:40 portfolio'
#traditional 60:40 portfolio weights (SPY: 60%, TLT: 40%) for performance benchmark
trad_wt = {'SPY': 0.6, 'TLT': 0.4, 'GLD': 0.0}
trad_wt_ser = pd.Series(trad_wt)[returns.columns]

'static risk parity'
#average rolling 60-day volatility across entire backtest period (used for static RP weights)
latest_vol = returns.rolling(window=60).std().mean()
inv_vol = 1 / latest_vol
risk_par_wt = inv_vol / inv_vol.sum()
risk_par_wt = risk_par_wt[returns.columns]

#function to calculate the contribution of each asset to the total portfolio volatility 
def calc_risk_dist(weights, cov_matrix):
    portfolio_vol = (weights @ cov_matrix @ weights.T) ** 0.5
    marginal_contrib = cov_matrix @ weights
    risk_contributions = weights * marginal_contrib / portfolio_vol
    return risk_contributions

#setting rolling volatility window, rebalance frequency, and number of data points
vol_wnd = 60
rebalance_freq = 21
n = len(returns)
dates = returns.index

#initialize containers for portfolio tracking
rp_dynamic_returns = []           #Portfolio returns (no transaction cost)
rp_dynamic_returns_cost = []      #Portfolio returns (with transaction cost)
risk_contribs_over_time = []      #Risk contribution per asset at rebalance
prev_wts = pd.Series(0, index=returns.columns)   #Last weights (for turnover calc)
turnover_record = []      #Turnover at each rebalance (for cost modeling)


for i in range(vol_wnd, n, rebalance_freq):
    wnd_returns = returns.iloc[i-vol_wnd:i]     #getting past 60-day asset returns for volatility estimation
    wts = compute_risk_parity_wt(wnd_returns)   #computing the risk parity weights for the rebalance 

    realized_vol = ((wnd_returns * wts).sum(axis=1)).std()  #calculate realized (historical) daily volatility of the portfolio over the past window
    leverage = target_annual_vol / realized_vol  if realized_vol!=0 else 0   #scale exposure so that the portfolio targets a specific annual volatility
    leverage = min(leverage,3)

    vix_today = vix_level.get(dates[i], 0)

    if vix_today>=20:    
        leverage *= 0.5
    
    if yield_spread.iloc[i] <= 0:
        leverage = leverage * 0.5

    cost_per_unit = 0.10                 #assumed transaction cost per unit of turnover
    wt_change = (wts - prev_wts).abs()   #calculating absolute change in portfolio weights since last rebalance
    turnover = wt_change.sum()           #total weight that needs to be traded
    turnover_record.append((dates[i], turnover))
    tc = turnover * cost_per_unit        #total transaction cost

    cov_matrix = wnd_returns.cov()       #computing asset return covariance matrix over the rolling window
    risk_contribs = calc_risk_dist(wts.values, cov_matrix.values)    #calculating each asset's contribution to total portfolio volatility
    print(f"\nDate: {dates[i].date()}")
    print("Weights:", dict(zip(returns.columns, wts.round(3))))
    print("Risk Contributions:", dict(zip(returns.columns, risk_contribs.round(3))))

    risk_contribs_over_time.append({
        'Date': dates[i],
        'GLD': risk_contribs[returns.columns.get_loc('GLD')],
        'SPY': risk_contribs[returns.columns.get_loc('SPY')],
        'TLT': risk_contribs[returns.columns.get_loc('TLT')]
    })

    for j in range(i, min(i+rebalance_freq, n)):   #holding period between rebalances — calculate daily returns using fixed weights
        gross_return = (returns.iloc[j] * wts).sum() * leverage   #gross return of each day 
        net_return = (gross_return - (tc / rebalance_freq))   #cost adjusted return of the day
        rp_dynamic_returns_cost.append((dates[j], net_return))  

        daily_return = (returns.iloc[j]*wts).sum() * leverage   
        rp_dynamic_returns.append((dates[j], daily_return))     #storing the daily return 

    prev_wts = wts.copy()

'realized (historical) volatility along with leverage of the static risk parity portfolio'
realized_vol_static = ((returns*risk_par_wt).sum()).std()
leverage_static = target_annual_vol/realized_vol_static if realized_vol_static!=0 else 0
risk_parity_returns = ((returns * risk_par_wt).sum(axis=1))*leverage_static  #daily returns of static risk parity portfolio, scaled by target leverage
cumulative_risk_par = (1+risk_parity_returns).cumprod() #total return of static risk parity


'60:40 portfolio returns'
trad_wt_par_returns = (returns * trad_wt_ser).sum(axis=1) #daily returns of the tradtional portfolio 
cumulative_trad = (1+trad_wt_par_returns).cumprod()  #total return of traditional portfolio



'creating dataframes of dynamic risk parity returns both daily and cumulative'

#creating a dataframe of dynamic risk parity daily returns (after transaction costs)
rp_dynamic_cost_df = pd.DataFrame(rp_dynamic_returns_cost, columns=['Date', 'Return']).set_index('Date')

#creating a dataframe of dynamic risk parity daily returns (no transaction costs)
rp_dynamic_df = pd.DataFrame(rp_dynamic_returns, columns=['Date', 'Return']).set_index('Date')

#calculating the cumulative performance of strategy with transaction costs included
cumulative_rp_dynamic_cost = (1 + rp_dynamic_cost_df['Return']).cumprod()

#calculating the cumulative performance of strategy with no transaction costs included
cumulative_rp_dynamic = (1 + rp_dynamic_df['Return']).cumprod()



#visualizing performance of three strategies: static RP, dynamic RP, and 60/40 benchmark
plt.figure(figsize=(12,6))
plt.plot(cumulative_risk_par, label='Static Risk Parity Portfolio')
plt.plot(cumulative_rp_dynamic_cost, label = 'Dynamic Risk Parity (With Cost)')
plt.plot(cumulative_trad, label = "60:40 Portfolio")
plt.title('Portfolio Performance Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

#zoom into COVID crash to evaluate performance
stress_strt = '2020-02-15'
stress_end = '2020-04-15'

risk_stress_stat = cumulative_risk_par[stress_strt:stress_end]
dyn_stress = cumulative_rp_dynamic_cost[stress_strt:stress_end]
trad_stress = cumulative_trad[stress_strt:stress_end]

#visualizing performance of three strategies: static RP, dynamic RP, and 60/40 benchmark during the Covid crash
plt.figure(figsize=(12,6))
plt.plot(risk_stress_stat, label='Risk Parity Portfolio')
plt.plot(dyn_stress, label = 'Dynamic Risk Parity (With Cost)')
plt.plot(trad_stress, label='60:40 Portfolio')
plt.title('Portfolio Performance During COVID Crash (2020)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()



#visualizing turnover to see how aggressively the dynamic portfolio is rebalancing over time
turnover_series = pd.DataFrame(turnover_record, columns=['Date', 'Turnover']).set_index('Date')


plt.figure(figsize=(12,6))
turnover_series['Turnover'].plot()
plt.title('Portfolio Turnover Over Time')
plt.xlabel('Date')
plt.ylabel('Turnover')
plt.grid(True)
plt.tight_layout()
plt.show()


'creating dataframe and plotting risk contribution of each asset in the portfolio'

#creating a dataframe of each asset's contribution to total portfolio risk at each rebalance
risk_contribs_df = pd.DataFrame(risk_contribs_over_time).set_index('Date')

#plotting stacked area chart of risk contributions over time
risk_contribs_df.plot.area(figsize=(12, 6), stacked=True)
plt.title("Risk Contribution Over Time")
plt.ylabel("Risk Contribution")
plt.xlabel("Date")
plt.legend(title="Assets")
plt.grid(True)
plt.tight_layout()
plt.show()

#computing annualized Sharpe ratio (risk-adjusted return)
def sharpe_ratio(returns, risk_free_rate = 0):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean()/excess_returns.std()

#computing maximum drawdown (largest peak-to-trough decline)
def max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    draw_down = (cumulative_returns - peak) / peak
    return draw_down.min()

#printing the performance metrics of the three strategies
print("\n📊 Performance Metrics:")
print(f"Dynamic Risk Parity Sharpe (With Cost): {sharpe_ratio(rp_dynamic_cost_df['Return']):.3f}")
print(f"Risk Parity Sharpe: {sharpe_ratio(risk_parity_returns):.3f}")
print(f"60/40 Sharpe: {sharpe_ratio(trad_wt_par_returns):.3f}")
print(f"Dynamic Risk Parity Max Drawdown: {max_drawdown(cumulative_rp_dynamic_cost):.2%}")
print(f"Risk Parity Max Drawdown: {max_drawdown(cumulative_risk_par):.2%}")
print(f"60/40 Max Drawdown: {max_drawdown(cumulative_trad):.2%}")

#calculating 6-month rolling Sharpe ratio (using 126 trading days) 
rolling_wnd = 126
rolling_sharpe = rp_dynamic_df['Return'].rolling(window = rolling_wnd).apply(lambda x:sharpe_ratio(x), raw = False)
rp_dynamic_df['Rolling_Sharpe'] = rolling_sharpe

#visualizing the 6-month rolling sharpe ratio
plt.figure(figsize=(12,6))
rolling_sharpe.plot()
plt.axhline(0, color='red', linestyle='--')
plt.title("6-Month Rolling Sharpe Ratio (Dynamic Risk Parity)")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.grid(True)
plt.show()


#compute 5-day forward return to label upcoming turbulence (used for ML target)
rp_dynamic_df['5_day_Forward_Return'] = rp_dynamic_df['Return'].pct_change(periods=5).shift(-5)

#label as '1' if next 5-day return drops more than 2% → indicates turbulence
rp_dynamic_df['Label'] = (rp_dynamic_df['5_day_Forward_Return'] < -0.02).astype(int)

#creating a dataframe of input features and target label for the ML model
features_df = pd.DataFrame({
    'VIX': vix_level,
    'Yield_Spread': yield_spread,
    'Rolling_Sharpe': rp_dynamic_df['Rolling_Sharpe'],
    'Turnover': turnover_series['Turnover'].reindex(rp_dynamic_df.index).ffill(),
    'Label': rp_dynamic_df['Label']
})

#inspecting feature integrity before model training
print("📋 Feature Snapshot:\n", features_df.head())
print("\n🔍 Feature Types:\n", features_df.dtypes)
print("\n🧼 Missing Values:\n", features_df.isnull().sum())

#drop rows where target label or rolling Sharpe is missing, then forward-fill remaining NaNs
features_clean = features_df.dropna(subset=['Rolling_Sharpe', 'Label'])
features_clean = features_clean.fillna(method='ffill')

print("\nAfter cleaning:")
print(features_clean.isnull().sum())
print("\nShape of cleaned dataset:", features_clean.shape)
features_clean.head()

features_clean[['VIX', 'Yield_Spread', 'Rolling_Sharpe', 'Turnover']].plot(subplots=True, figsize=(12, 10), title='Feature Time Series')
plt.tight_layout()
plt.show()

features_clean['Label'].value_counts().plot(kind='bar', title='Label Distribution (0 = Calm, 1 = Turbulent)')
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(True)
plt.show()

#feature matrix (X) and target variable (Y) for ML classification
X = features_clean[['VIX', 'Yield_Spread', 'Rolling_Sharpe', 'Turnover']]
Y = features_clean['Label']

#use time-series split (no shuffling) to preserve temporal structure for backtesting-style validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle = False)

#train a Random Forest with 200 trees and limited depth to avoid overfitting
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, Y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

print("\n📊 Classification Report:\n", classification_report(Y_test, y_pred))
print("Confusion Matrix:")
sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#predicting turbulence probablilites across the entire dataset 
all_proba = rf.predict_proba(X)[:, 1]
turbulence_prob = pd.Series(all_proba, index=X.index, name="Turbulence_Prob")

##initialize containers for ML portfolio tracking 
rp_ml_returns = []
rp_ml_cost_returns = []
prev_wts = pd.Series(0, index=returns.columns)
ml_turnover = []

for i in range(vol_wnd, n, rebalance_freq):
    wnd_returns = returns.iloc[i-vol_wnd:i]
    wts = compute_risk_parity_wt(wnd_returns)

    realized_vol = ((wnd_returns * wts).sum(axis=1)).std()
    leverage = target_annual_vol / realized_vol if realized_vol != 0 else 0
    leverage = min(leverage, 3)

    proba_today = turbulence_prob.get(dates[i], 0)
    if proba_today > 0.8:
        leverage *= 0.25
    elif proba_today > 0.6:
        leverage *= 0.5

    wt_change = (wts - prev_wts).abs()
    turnover = wt_change.sum()
    ml_turnover.append((dates[i], turnover))
    tc = turnover * cost_per_unit

    for j in range(i, min(i + rebalance_freq, n)):
        gross = (returns.iloc[j] * wts).sum() * leverage
        net = gross - (tc / rebalance_freq)

        rp_ml_returns.append((dates[j], gross))
        rp_ml_cost_returns.append((dates[j], net))

    prev_wts = wts.copy()

ml_cost_df = pd.DataFrame(rp_ml_cost_returns, columns=['Date', 'Return']).set_index('Date')
cumulative_ml = (1 + ml_cost_df['Return']).cumprod()

plt.figure(figsize=(12,6))
plt.plot(cumulative_risk_par, label='Static RP')
plt.plot(cumulative_rp_dynamic_cost, label='Dynamic RP (VIX+Spread)')
plt.plot(cumulative_ml, label='Dynamic RP (ML-based)')
plt.plot(cumulative_trad, label='60/40 Portfolio')
plt.title("Final Strategy Comparison (incl. ML-based RP)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()   



#visualizing the most predictive features in the model
importances = rf.feature_importances_
feat_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feat_names)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(True)
plt.tight_layout()
plt.show()
print("Accuracy:", accuracy_score(Y_test, y_pred))

auc = roc_auc_score(Y_test, y_proba)
print(f"ROC-AUC Score: {auc:.3f}")
fpr, tpr, _ = roc_curve(Y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

cumulative_ml.to_csv("ml_strategy_returns.csv")