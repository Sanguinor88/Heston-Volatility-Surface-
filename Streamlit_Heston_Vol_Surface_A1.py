import streamlit as st
import yfinance as yf 
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
from scipy.integrate import quad
import plotly.graph_objects as go

st.title('Implied Volatility Surface Area')

def heston_call_price(S, K, T, r, v0, kappa, theta, sigma, rho, q=0):
    def integrand(phi, Pnum):
        i = complex(0, 1)  
        u = 0.5 if Pnum == 1 else -0.5
        b = kappa + u * rho * sigma
        a = kappa * theta
        d = np.sqrt((rho * sigma * i * phi - b) ** 2 - sigma ** 2 * (2 * u * i * phi - phi ** 2))
        g = (b - rho * sigma * i * phi + d) / ((b - rho * sigma * i * phi - d) + 1e-8)

        C = (r - q) * i * phi * T + (a / sigma ** 2) * ((b - rho * sigma * i * phi + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
        D = ((b - rho * sigma * i * phi + d) / sigma ** 2) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
        f = np.exp(C + D * v0 + i * phi * np.log(S * np.exp(-q * T)))
        return np.real(np.exp(-i * phi * np.log(K)) * f / (i * phi))

    P1 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 1), 0, np.inf, limit=100)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 2), 0, np.inf, limit=100)[0]

    call_price = np.exp(-q * T) * S * P1 - np.exp(-r * T) * K * P2
    return call_price

def implied_volatility(S, K, T, r, price, v0, kappa, theta, sigma, rho, q):
    if T <= 0 or price <= 0:
        return np.nan

    def objective_function(sigma):
        return heston_call_price(S, K, T, r, v0, kappa, theta, sigma, rho, q) - price

    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
    except ValueError:
        implied_vol = np.nan

    return implied_vol

st.sidebar.header('Input Parameters')
ticker_symbol = st.sidebar.text_input('Enter Ticker', value='SPY', max_chars=10).upper()

Risk_Free_Rate = st.sidebar.number_input('Risk Free Rate (eg, 0.04 for 4%)', value=0.04, format="%.4f")
dividend_yield = st.sidebar.number_input("Dividend Yield (eg, 0.04 for 4%)", value=0.02, format="%.4f")

st.sidebar.header('Heston Model Parameters')
kappa = st.sidebar.number_input("Speed of Mean Reversion (kappa)", value=1.5, step=0.1)
theta = st.sidebar.number_input("Long-Term Variance (theta)", value=0.04, step=0.01)
sigma = st.sidebar.number_input("Volatility of Variance (sigma)", value=0.2, step=0.01)
rho = st.sidebar.number_input("Correlation (rho)", value=-0.7, step=0.1)

st.sidebar.header('Strike Price Inputs')
min_strike_price_pct = st.sidebar.number_input('Minimum Strike Price %', min_value=10.00, max_value=499.00, value=80.00, step=1.0, format="%.1f")
max_strike_price_pct = st.sidebar.number_input('Maximum Strike Price %', min_value=11.0, max_value=500.00, value=130.00, step=1.0, format="%.1f")

ticker = yf.Ticker(ticker_symbol)

try:
    spot_history = ticker.history(period='1y')  
    spot_price = spot_history['Close'].dropna().iloc[-1]
except Exception as e:
    st.error(f"An error occurred while fetching spot price data: {e}")
    st.stop()

try:
    expirations = ticker.options
except Exception as e:
    st.error(f'Failed to fetch options data for {ticker_symbol}: {e}')
    st.stop()

option_data = []
for exp_date in expirations:
    try:
        opt_chain = ticker.option_chain(exp_date)
        calls = opt_chain.calls
        calls = calls[(calls['bid'].fillna(0) > 0) & (calls['ask'].fillna(0) > 0)]
        for _, row in calls.iterrows():
            option_data.append({
                'expirationDate': pd.Timestamp(exp_date),
                'strike': row['strike'],
                'mid': (row['bid'] + row['ask']) / 2
            })
    except Exception as e:
        continue

options_df = pd.DataFrame(option_data)
options_df['timeToExpiration'] = (options_df['expirationDate'] - pd.Timestamp.today()).dt.days / 365

options_df['hestonPrice'] = options_df.apply(lambda row: heston_call_price(spot_price, row['strike'], row['timeToExpiration'], Risk_Free_Rate, theta, kappa, theta, sigma, rho, dividend_yield), axis=1)

fig = go.Figure(data=[go.Surface(
    x=options_df['timeToExpiration'],
    y=options_df['strike'],
    z=options_df['hestonPrice'],
    colorscale='Viridis'
)])

fig.update_layout(title=f'Heston Model Volatility Surface for {ticker_symbol}',
                  scene=dict(xaxis_title='Time to Expiration', yaxis_title='Strike Price', zaxis_title='Option Price'))

st.plotly_chart(fig)

st.markdown("By Stephen Chen & Jack Armstrong")
