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
        """Characteristic function integrand."""
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
st.sidebar.write('Heston Model Adjustments.')

st.sidebar.header('Ticker')
ticker_symbol = st.sidebar.text_input(
    'Enter Ticker',
    value='SPY',
    max_chars=10
).upper()

Risk_Free_Rate = st.sidebar.number_input(
    'Risk Free Rate (eg, 0.04 for 4%)',
    value=0.04,
    format="%.4f"
)

dividend_yield = st.sidebar.number_input(
    "Dividend Yield (eg, 0.04 for 4%)",
    value=0.02,
    format="%.4f"
)

st.sidebar.header('Heston Model Parameters')
kappa = st.sidebar.number_input("Speed of Mean Reversion (kappa)", value=1.5, step=0.1)
theta = st.sidebar.number_input("Long-Term Variance (theta)", value=0.04, step=0.01)
sigma = st.sidebar.number_input("Volatility of Variance (sigma)", value=0.2, step=0.01)
rho = st.sidebar.number_input("Correlation (rho)", value=-0.7, step=0.1)

st.sidebar.header('Strike Price Inputs')
min_strike_price_pct = st.sidebar.number_input(
    'Minimum Strike Price %',
    min_value=10.00,
    max_value=499.00,
    value=80.00,
    step=1.0,
    format="%.1f"
)
max_strike_price_pct = st.sidebar.number_input(
    'Maximum Strike Price %',
    min_value=11.0,
    max_value=500.00,
    value=130.00,
    step=1.0,
    format="%.1f"
)

if min_strike_price_pct >= max_strike_price_pct:
    st.sidebar.error('Minimum percentage must be less than the maximum.')
    st.stop()

ticker = yf.Ticker(ticker_symbol)

today = pd.Timestamp('today').normalize()

try:
    spot_history = ticker.history(period='1y')  
    if spot_history.empty:
        st.error("Failed to retrieve historical spot price data.")
        st.stop()
    log_returns = np.log(spot_history['Close'] / spot_history['Close'].shift(1))
    v0 = np.var(log_returns)  
except Exception as e:
    st.error(f"Error calculating initial variance: {e}")
    st.stop()

try:
    expirations = ticker.options
except Exception as e:
    st.error(f'No available data for {ticker_symbol}.')
    st.stop()

option_data = []

for exp_date in expirations:
    try:
        opt_chain = ticker.option_chain(exp_date)
        calls = opt_chain.calls
        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]

        for index, row in calls.iterrows():
            strike = row['strike']
            bid = row['bid']
            ask = row['ask']
            mid_price = (bid + ask) / 2

            option_data.append({
                'expirationDate': pd.Timestamp(exp_date),
                'strike': strike,
                'bid': bid,
                'ask': ask,
                'mid': mid_price
            })
    except Exception as e:
        st.warning(f'Option chain collection failure for {exp_date}: {e}')
        continue

if not option_data:
    st.error('No option data available after filtering.')
    st.stop()

options_df = pd.DataFrame(option_data)

try:
    spot_price = spot_history['Close'].iloc[-1]
except Exception as e:
    st.error(f'An error occurred while fetching spot price data: {e}')
    st.stop()

# Validate 'expirationDate' column
if 'expirationDate' not in options_df.columns:
    st.error("'expirationDate' column is missing. Unable to calculate time to expiration.")
    st.stop()

# Calculate 'daysToExpiration'
try:
    options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
except Exception as e:
    st.error(f"Error calculating 'daysToExpiration': {e}")
    st.stop()

# Check for invalid or missing 'daysToExpiration'
if options_df['daysToExpiration'].isna().any() or (options_df['daysToExpiration'] <= 0).all():
    st.error("Invalid or missing 'daysToExpiration'. Ensure expiration dates are valid and in the future.")
    st.stop()

# Calculate 'timeToExpiration'
try:
    options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365
except Exception as e:
    st.error(f"Error calculating 'timeToExpiration': {e}")
    st.stop()

# Validate 'timeToExpiration'
if options_df['timeToExpiration'].isna().any() or options_df['timeToExpiration'].empty:
    st.error("Invalid or missing 'timeToExpiration'. Unable to continue.")
    st.stop()

# Calculate moneyness
options_df['moneyness'] = options_df['strike'] / spot_price

# Validate moneyness column
if options_df['moneyness'].isna().any() or options_df['moneyness'].empty:
    st.error("Moneyness values are invalid or missing. Unable to continue.")
    st.stop()

# Ensure moneyness values are valid
moneyness_min = options_df['moneyness'].min()
moneyness_max = options_df['moneyness'].max()

# Debugging logs to check the issue
st.write("Moneyness Min:", moneyness_min)
st.write("Moneyness Max:", moneyness_max)

# Check if moneyness range is valid
if pd.isna(moneyness_min) or pd.isna(moneyness_max):
    st.error("Moneyness range is invalid. Unable to create bins.")
    st.stop()

if moneyness_min == moneyness_max:
    st.warning("All moneyness values are identical. Using default fallback bins.")
    moneyness_bins = [moneyness_min - 0.05, moneyness_min, moneyness_min + 0.05]
else:
    moneyness_bins = np.linspace(moneyness_min, moneyness_max, 4)

# Ensure bins are strictly increasing
if len(set(moneyness_bins)) < len(moneyness_bins):
    st.warning("Bins have overlapping or identical values. Expanding range slightly.")
    moneyness_bins = np.linspace(moneyness_min - 0.01, moneyness_max + 0.01, 4)

# Add tranches
try:
    tranches = ['Low', 'Mid', 'High']
    options_df['tranche'] = pd.cut(
        options_df['moneyness'], 
        bins=moneyness_bins, 
        labels=tranches, 
        include_lowest=True
    )
except ValueError as e:
    st.error(f"Error in assigning tranches: {e}")
    st.stop()

# Ensure options_df is not empty
if options_df.empty:
    st.error("No valid options data available for Heston model calculations. Please adjust your inputs.")
    st.stop()

# Validate required columns
required_columns = ['strike', 'timeToExpiration']
missing_columns = [col for col in required_columns if col not in options_df.columns]
if missing_columns:
    st.error(f"Missing required columns for Heston model calculations: {missing_columns}")
    st.stop()

# Debug: Print the first few rows of options_df
st.write("Options DataFrame (first few rows):")
st.write(options_df.head())

# Function to safely calculate Heston call price
def safe_heston_call_price(row):
    try:
        return heston_call_price(
            S=spot_price,
            K=row['strike'],
            T=row['timeToExpiration'],
            r=Risk_Free_Rate,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            q=dividend_yield
        )
    except Exception as e:
        st.warning(f"Error calculating hestonPrice for strike={row['strike']}, T={row['timeToExpiration']}: {e}")
        return np.nan

# Filter out rows with very short time to expiration
filtered_options_df = options_df[options_df['timeToExpiration'] > 0.01]

# Check if there are valid rows remaining
if filtered_options_df.empty:
    st.error("No valid options data after filtering for time to expiration. Please check your input parameters.")
    st.stop()

# Calculate hestonPrice with error handling
with st.spinner('Calculating option prices using Heston model...'):
    filtered_options_df['hestonPrice'] = filtered_options_df.apply(safe_heston_call_price, axis=1)

# Drop rows with NaN hestonPrice
filtered_options_df.dropna(subset=['hestonPrice'], inplace=True)

# Check if hestonPrice calculation succeeded
if filtered_options_df.empty:
    st.error("All Heston model calculations failed. Please check your input parameters or model configuration.")
    st.stop()

# Update options_df with successful calculations
options_df = filtered_options_df

# Calculate tranche summary
try:
    tranche_summary = options_df.groupby('tranche').agg(
        Low=('hestonPrice', 'min'),
        Mid=('hestonPrice', 'mean'),
        High=('hestonPrice', 'max')
    ).reset_index()

    st.write("### Low, Mid, and High Prices by Tranche")
    st.table(tranche_summary)
except KeyError as e:
    st.error(f"Error in calculating tranche summary: {e}")
    st.stop()

# Render 3D Volatility Surface Chart
if len(options_df) > 0:
    # Prepare data for the 3D chart
    Y = options_df['strike'].values
    X = options_df['timeToExpiration'].values
    Z = options_df['hestonPrice'].values

    if len(X) > 0 and len(Y) > 0 and len(Z) > 0:
        # Create the 3D chart
        ti = np.linspace(X.min(), X.max(), 50)
        ki = np.linspace(Y.min(), Y.max(), 50)
        T, K = np.meshgrid(ti, ki)

        # Interpolate grid for the surface
        Zi = griddata((X, Y), Z, (T, K), method='linear')
        Zi = np.ma.array(Zi, mask=np.isnan(Zi))

        fig = go.Figure(data=[go.Surface(
            x=T, y=K, z=Zi,
            colorscale='Viridis',
            colorbar_title='Option Price ($)'
        )])

        fig.update_layout(
            title=f'Heston Model Option Price Surface for {ticker_symbol}',
            scene=dict(
                xaxis_title='Time to Expiration (years)',
                yaxis_title='Strike Price ($)',
                zaxis_title='Option Price ($)'
            ),
            autosize=False,
            width=900,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        st.plotly_chart(fig)
    else:
        st.error("Insufficient data to plot the volatility surface. Please check your input parameters.")
else:
    st.error("No valid data available to generate the volatility surface.")

st.write("---")
st.markdown(
    "By Stephen Chen & Jack Armstrong | linkedin.com/in/stephen-chen-60b2b3184 & linkedin.com/in/jack-armstrong-094932241"
)
