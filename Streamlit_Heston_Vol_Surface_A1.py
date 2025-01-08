import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad
import plotly.graph_objects as go

# Title
st.title("Implied Volatility Surface Area")

# Sidebar Input Parameters
st.sidebar.header("Input Parameters")
st.sidebar.write("Heston Model Adjustments.")

ticker_symbol = st.sidebar.text_input(
    "Enter Ticker", value="SPY", max_chars=10
).upper()

Risk_Free_Rate = st.sidebar.number_input(
    "Risk-Free Rate (e.g., 0.04 for 4%)", value=0.04, format="%.4f"
)

# Strike Price Input
st.sidebar.header("Strike Price Inputs")
min_strike_price_pct = st.sidebar.number_input(
    "Minimum Strike Price %",
    min_value=10.00,
    max_value=499.00,
    value=80.00,
    step=1.0,
    format="%.1f"
)
max_strike_price_pct = st.sidebar.number_input(
    "Maximum Strike Price %",
    min_value=11.0,
    max_value=500.00,
    value=130.00,
    step=1.0,
    format="%.1f"
)

if min_strike_price_pct >= max_strike_price_pct:
    st.sidebar.error("Minimum percentage must be less than the maximum.")
    st.stop()

# Heston Model Parameters (set default values)
kappa = 1.6
theta = 0.04
sigma = 0.2
rho = -0.7

# Fetch Spot Price and Options Data
def fetch_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    
    # Fetch spot price
    try:
        spot_history = ticker.history(period="1d")
        spot_price = spot_history["Close"].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching spot price: {e}")
        return None, None

    # Fetch options data
    try:
        expirations = ticker.options
        options_data = []

        for exp_date in expirations:
            opt_chain = ticker.option_chain(exp_date)
            calls = opt_chain.calls
            calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]

            for _, row in calls.iterrows():
                mid_price = (row["bid"] + row["ask"]) / 2
                options_data.append({
                    "expirationDate": pd.Timestamp(exp_date),
                    "strike": row["strike"],
                    "bid": row["bid"],
                    "ask": row["ask"],
                    "mid": mid_price,
                })

        return spot_price, pd.DataFrame(options_data)
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        return None, None

# Calculate Greeks
def calculate_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    return delta, gamma, theta, vega, rho

# Main Code Execution
spot_price, options_df = fetch_data(ticker_symbol)

if spot_price and not options_df.empty:
    today = pd.Timestamp("today").normalize()

    # Add Time to Expiration
    options_df["daysToExpiration"] = (options_df["expirationDate"] - today).dt.days
    options_df = options_df[options_df["daysToExpiration"] > 0]
    options_df["timeToExpiration"] = options_df["daysToExpiration"] / 365

    # Filter options by strike price range
    min_strike = spot_price * (min_strike_price_pct / 100)
    max_strike = spot_price * (max_strike_price_pct / 100)
    options_df = options_df[(options_df["strike"] >= min_strike) & (options_df["strike"] <= max_strike)]

    # Calculate Greeks for each option
    greeks = []
    for _, row in options_df.iterrows():
        try:
            delta, gamma, theta, vega, rho = calculate_greeks(
                S=spot_price,
                K=row["strike"],
                T=row["timeToExpiration"],
                r=Risk_Free_Rate,
                sigma=sigma,
            )
            greeks.append({
                "strike": row["strike"],
                "expirationDate": row["expirationDate"],
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "rho": rho,
            })
        except Exception as e:
            st.warning(f"Error calculating Greeks for strike={row['strike']}: {e}")

    # Display Greeks Table
    greeks_df = pd.DataFrame(greeks)
    st.write("### Options Greeks")
    st.dataframe(greeks_df.style.format(precision=4).set_table_attributes("style='display:inline'"))

    # Display Options Prices Table
    st.write("### Options Prices")
    st.dataframe(options_df[["strike", "expirationDate", "bid", "ask", "mid"]])

    # 3D Surface Plot for Greeks (Delta as an Example)
    X = greeks_df["strike"].values
    Y = greeks_df["expirationDate"].map(lambda x: (x - today).days).values
    Z = greeks_df["delta"].values

    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z, colorscale="Viridis", colorbar_title="Delta"
    )])
    fig.update_layout(
        title=f"Delta Surface for {ticker_symbol}",
        scene=dict(
            xaxis_title="Strike Price ($)",
            yaxis_title="Days to Expiration",
            zaxis_title="Delta",
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
