import pandas as pd
import numpy as np
import yfinance as yf
import math
import os
from scipy.optimize import minimize
from pathlib import Path
import sys

# avoid traceback error
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#import the black litterman portfolio python file
import black_litterman.black_litterman_portfolio_12months_rolling as blo

# LIST OF FUNCTIONS FOR THE RISK DASHBOARD

# Historical VaR and CVaR (95%)

def historical_var(r, level = 5):

    if isinstance(r, pd.DataFrame):
        return r.aggregate(historical_var,level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("r is expected to be a Series or DataFrame")
    
def historical_cvar(r, level = 5):

    if isinstance(r, pd.Series):
        is_beyond = r <= -historical_var(r, level=level)
        return -r[is_beyond].mean()

    elif isinstance(r, pd.DataFrame):
        return r.aggregate(historical_cvar, level=level)

    else:
        raise TypeError("Expected r to be a Series or DataFrame")

# Annualized portfolio volatility

def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights) ** 0.5

# Annualized sharpe ratio

def annualize_rets(r, periods_per_year):

    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]

    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualize_vol(r, periods_per_year):

    return r.std() * (periods_per_year ** 0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns.
    """
    # Convert the annual riskfree rate to per period
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period

    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)

    if ann_vol == 0:
        return np.nan

    return ann_ex_ret / ann_vol

# Annualized sortino ratio

def sortino_ratio(r, riskfree_rate=0.0, periods_per_year=12):
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period

    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)

    downside = excess_ret[excess_ret < 0]
    downside_dev = downside.std(ddof=0) * (periods_per_year ** 0.5)

    if downside_dev == 0:
        return np.nan

    return ann_ex_ret / downside_dev

# Annualized skewness

def skewness(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)

    exp = (demeaned_r ** 3).mean()

    return exp / sigma_r ** 3

def kurtosis(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)

    exp = (demeaned_r ** 4).mean()

    return exp / sigma_r ** 4

# test to see if the code works (final rolling period)
#BLOvar_95 = historical_var(blo.valid_net_returns,level=5)
#BLOcvar_95 = historical_cvar(blo.valid_net_returns,level=5)
#print(f"Historical 95% VAR: {BLOvar_95:.2%}")
#print(f"Historical 95% CVAR: {BLOcvar_95:.2%}")

# main body code

rolling_window = 12 # 12 month rolling period
initial_wealth = 1000 # $1,000 assumed as the initial wealth for the wealth curve and dropdown

# historical VaR and CVaR for rolling net returns
rolling_BLOvar_95 = blo.valid_net_returns.rolling(window=rolling_window, min_periods=rolling_window).apply(
    lambda x: historical_var(pd.Series(x), level=5), raw=False
)

rolling_BLOcvar_95 = blo.valid_net_returns.rolling(window=rolling_window,min_periods=rolling_window).apply(
    lambda x: historical_cvar(pd.Series(x), level=5), raw=False
)

# rolling annualized portfolio volatility
rolling_portfolio_vol = blo.valid_net_returns.rolling(window=rolling_window,
 min_periods=rolling_window).std() * (rolling_window ** 0.5)

# rolling annualized sharpe ratio
rolling_sharpe_ratio = blo.valid_net_returns.rolling(window=rolling_window,min_periods=rolling_window).apply(
    lambda x: sharpe_ratio(x,blo.risk_free["Risk_Free_Rate"].asof(x.index[-1]),rolling_window), raw=False
)

# rolling annualized sortino ratio
rolling_sortino_ratio = blo.valid_net_returns.rolling(window=rolling_window,min_periods=rolling_window).apply(
    lambda x: sortino_ratio(x,blo.risk_free["Risk_Free_Rate"].asof(x.index[-1]),rolling_window), raw=False
)

# rolling skewness
rolling_skewness = blo.valid_net_returns.rolling(window=rolling_window,min_periods=rolling_window).apply(
    lambda x: skewness(x), raw=False
)

# rolling kurtosis
rolling_kurtosis = blo.valid_net_returns.rolling(window=rolling_window,min_periods=rolling_window).apply(
    lambda x: kurtosis(x), raw=False
)

# rolling correlation (with benchmark)
aligned_return = pd.concat([blo.valid_net_returns, blo.valid_benchmark_returns],axis=1,join='inner')
aligned_return.columns=["Portfolio","Benchmark"]

rolling_correlation = aligned_return["Portfolio"].rolling(window=rolling_window,
         min_periods=rolling_window).corr(aligned_return["Benchmark"])

active_return = aligned_return["Portfolio"] - aligned_return["Benchmark"]

# rolling tracking error
rolling_tracking_error = active_return.rolling(window=rolling_window,
        min_periods=rolling_window).std() * (rolling_window**0.5)

# wealth curve
wealth_curve = initial_wealth * (1 + blo.valid_net_returns).cumprod()
benchmark_wealth_curve = initial_wealth * (1 + aligned_return["Benchmark"]).cumprod()
portfolio_wealth_curve = initial_wealth * (1 + aligned_return["Portfolio"]).cumprod()

wealth_curves = pd.DataFrame({"Portfolio Wealth": portfolio_wealth_curve, 
                              "Benchmark Wealth": benchmark_wealth_curve})


# drawdown
previous_peak = wealth_curve.cummax()
drawdown = (wealth_curve - previous_peak) / previous_peak
max_drawdown_date = drawdown.idxmin()
max_drawdown = drawdown.min()
current_drawdown = drawdown.iloc[-1]

# time to recovery
peak_before_drawdown_date = wealth_curve.loc[:max_drawdown_date].idxmax()
peak_before_drawdown_value = wealth_curve.loc[peak_before_drawdown_date]

recovery = wealth_curve.loc[max_drawdown_date:]
recovery_dates = recovery[recovery >= peak_before_drawdown_value]

if not recovery_dates.empty:
    recovery_date = recovery_dates.index[0]
    time_to_recovery = (recovery_date.to_period("M") - max_drawdown_date.to_period("M")).n
else:
    recovery_date = None
    time_to_recovery = None

time_to_recovery_info = pd.DataFrame({"Metric" : [
        "Peak Before Max Drawdown","Max Drawdown Date","Max Drawdown","Recovery Date",
        "Time to Recovery (Months)"
    ],"Value": [peak_before_drawdown_date,max_drawdown_date,max_drawdown,recovery_dates,
                time_to_recovery] })

print(time_to_recovery_info)

# more drawdown info
worst_month = blo.valid_net_returns.idxmin()
worst_month_return = blo.valid_net_returns.min()
worst_month_info = pd.DataFrame({"Worst Month": [worst_month],
                                 "Worst Month's Return": [worst_month_return]})
#print(worst_month_info)

print('works')
'''
rolling_tail_risk = pd.DataFrame({
    "Rolling Historical VaR 95%": rolling_BLOvar_95,
    "Rolling Historical CVaR 95%": rolling_BLOcvar_95
})

print(rolling_tail_risk.tail())
'''