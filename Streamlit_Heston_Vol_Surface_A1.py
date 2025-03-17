import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go

st.title(" Heston Model Implied Volatility Surface Area")

st.sidebar.header("Input Parameters")
st.sidebar.write("Heston Model Adjustments.")

ticker_symbol = st.sidebar.text_input(
    "Enter Ticker", value="SPY", max_chars=10
).upper()

Risk_Free_Rate = st.sidebar.number_input(
    "Risk-Free Rate (e.g., 0.04 for 4%)", value=0.04, format="%.4f"
    )

st.sidebar.header("Strike Price Inputs")
min_strike_price_pct = st.sidebar.number_input(
    "Minimum Strike Price",
    min_value=10.00,
    max_value=499.00,
    value=80.00,
    step=1.0,
    format="%.1f"
)
max_strike_price_pct = st.sidebar.number_input(
    "Maximum Strike Price",
    min_value=11.0,
    max_value=500.00,
    value=130.00,
    step=1.0,
    format="%.1f"
)

if min_strike_price_pct >= max_strike_price_pct:
    st.sidebar.error("Minimum percentage must be less than the maximum.")
    st.stop()

kappa = 1.6
theta = 0.04
sigma = 0.2
rho = -0.7

def fetch_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        spot_history = ticker.history(period="1d")
        if spot_history.empty or "Close" not in spot_history.columns:
            st.error("Failed to retrieve spot price. The specified ticker may be invalid, or the API may be experiencing downtime.")
            return None, None
        spot_price = spot_history["Close"].dropna()
        if spot_price.empty:
            st.error("Failed to retrueve spot price data is empty.")
            return None, None
        spot_price = spot_price.iloc[-1]
    except Exception as e:
        st.error(f"An error occurred while fetching spot price data: {e}")
        return None, None

    try:
        expirations = ticker.options
        if not expirations:
            st.error(f"No options data available for {ticker_symbol}. The ticker may not have options or API issues may exist.")
            return spot_price, None
        option_data = []

        for exp_date in expirations:
            try:
                opt_chain = ticker.option_chain(exp_date)
                calls = opt_chain.calls
                if calls.empty:
                    continue
                calls = calls[(calls["bid"].fillna(0) > 0) & (calls["ask"].fillna(0) > 0)]
                if calls.empty:
                    continue
                for _, row in calls.iterrows():
                    mid_price = (row["bid"] + row["ask"]) / 2
                    option_data.append({
                        "expirationDate": pd.Timestamp(exp_date),
                        "strike": row["strike"],
                        "bid": row["bid"],
                        "ask": row["ask"],
                        "mid": mid_price,
                    })
            except Exception as e:
                st.warning(f"Failed to fetch option chain for {exp_date}: {e}")
                continue
        if not option_data:
            st.error("No valid options data available after filtering.")
            return spot_price, None
        
        return spot_price, pd.DataFrame(option_data)
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        return None, None

spot_price, options_df = fetch_data(ticker_symbol)

if spot_price and options_df is not None and not options_df.empty:
    today = pd.Timestamp("today").normalize()

    options_df["daysToExpiration"] = (options_df["expirationDate"] - today).dt.days
    options_df = options_df[options_df["daysToExpiration"] > 0]
    options_df["timeToExpiration"] = options_df["daysToExpiration"] / 365

    min_strike = spot_price * (min_strike_price_pct / 100)
    max_strike = spot_price * (max_strike_price_pct / 100)
    options_df = options_df[(options_df["strike"] >= min_strike) & (options_df["strike"] <= max_strike)]

    if options_df.empty:
        st.error("No options available in the specified strike range.")
        st.stop()

    X = options_df["strike"].values
    Y = options_df["timeToExpiration"].values
    Z = options_df["mid"].values

    xi = np.linspace(X.min(), X.max(), 50)
    yi = np.linspace(Y.min(), Y.max(), 50)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((X, Y), Z, (xi, yi), method="linear")

    fig = go.Figure(data=[go.Surface(
        x=xi, y=yi, z=zi, colorscale="Viridis", colorbar_title="Mid Price"
    )])
    fig.update_layout(
        title=f"Implied Volatility Surface for {ticker_symbol}",
        scene=dict(
            xaxis_title="Strike Price ($)",
            yaxis_title="Time to Maturity (Years)",
            zaxis_title="Mid Price"
        ),
        autosize=False,
        width=900,
        height=800,
    )

    st.plotly_chart(fig)
else:
    st.error("Failed to retrieve valid options data. Please check your inputs.")

st.write("---")
st.markdown(
    "By Stephen Chen & Jack Armstrong | linkedin.com/in/stephen-chen-60b2b3184 & linkedin.com/in/jack-armstrong-094932241"
)
