import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Efficient Frontier Simulator", layout="wide")

st.title('Efficient Frontier Simulator with Constraints')

# --- INPUTS ---
# Asset names
assets = st.text_input('Enter Asset Names (comma-separated)',
                       'Stocks, Bonds, Real Estate').split(',')
assets = [asset.strip() for asset in assets]
num_assets = len(assets)

st.subheader('Asset Assumptions')

cols = st.columns(num_assets)
expected_returns = []
volatilities = []
max_weights = []

for idx, col in enumerate(cols):
    with col:
        expected_returns.append(
            st.slider(f'{assets[idx]} Return (%)', -5.0, 20.0, 6.0, step=0.1))
        volatilities.append(
            st.slider(f'{assets[idx]} Volatility (%)',
                      1.0,
                      50.0,
                      15.0,
                      step=0.5))
        max_weights.append(
            st.slider(f'{assets[idx]} Max Weight (%)',
                      0.0,
                      100.0,
                      100.0,
                      step=1.0))

# Correlation Matrix Input
st.subheader('Correlation Matrix')


def correlation_input():
    cor_matrix = np.identity(num_assets)
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            cor_val = st.slider(f'Correlation {assets[i]} - {assets[j]}',
                                -1.0,
                                1.0,
                                0.2,
                                step=0.05)
            cor_matrix[i, j] = cor_val
            cor_matrix[j, i] = cor_val
    return cor_matrix


correlation_matrix = correlation_input()

# --- CALCULATIONS ---
returns = np.array(expected_returns) / 100
stds = np.array(volatilities) / 100
max_weights = np.array(max_weights) / 100

cov_matrix = np.outer(stds, stds) * correlation_matrix

# Simulate portfolios
num_portfolios = 5000
results = []
weights_record = []

for _ in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    # Apply max weight constraints
    if np.any(weights > max_weights):
        continue

    port_return = np.dot(weights, returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_volatility
    results.append([port_return, port_volatility, sharpe])
    weights_record.append(weights)

results = np.array(results)
weights_record = np.array(weights_record)

# --- OUTPUTS ---

st.subheader('Efficient Frontier')
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(results[:, 1],
                     results[:, 0],
                     c=results[:, 2],
                     cmap='viridis')
fig.colorbar(scatter, label='Sharpe Ratio')
ax.set_xlabel('Volatility (Risk)')
ax.set_ylabel('Expected Return')
ax.set_title('Efficient Frontier')
st.pyplot(fig)

# Best Portfolio
if len(results) > 0:
    best_idx = np.argmax(results[:, 2])
    st.subheader('Best Portfolio (Highest Sharpe)')
    best_weights = weights_record[best_idx]
    best_return = results[best_idx, 0]
    best_risk = results[best_idx, 1]
    best_sharpe = results[best_idx, 2]

    st.metric(label="Expected Return", value=f"{best_return*100:.2f}%")
    st.metric(label="Risk (Volatility)", value=f"{best_risk*100:.2f}%")
    st.metric(label="Sharpe Ratio", value=f"{best_sharpe:.2f}")

    best_df = pd.DataFrame({'Asset': assets, 'Weight': best_weights})
    st.dataframe(best_df.style.format({"Weight": "{:.2%}"}))
else:
    st.warning(
        'No portfolios met the maximum weight constraints. Try adjusting them.'
    )

# Download full portfolios
if len(results) > 0:
    st.subheader('Download Portfolios')
    portfolios_df = pd.DataFrame(weights_record, columns=assets)
    portfolios_df['Expected Return'] = results[:, 0]
    portfolios_df['Risk (Volatility)'] = results[:, 1]
    portfolios_df['Sharpe Ratio'] = results[:, 2]

    csv = portfolios_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Portfolios CSV",
                       data=csv,
                       file_name='efficient_frontier_portfolios.csv',
                       mime='text/csv')
