import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from usr_func.replicate_signals import replicate_signals
from usr_func.std_dev_to_prob import std_dev_to_prob

st.sidebar.title("Statistical Limit Analysis for anomaly detection")
uploaded_file = st.sidebar.file_uploader("DataSet", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.DataFrame()

noise_level = st.sidebar.slider("Noise Level", 0.01, 2.0, 0.1, 0.01)
time_shift_range = st.sidebar.slider("Time Shift Range", 0, 100, 20, 1)
std_deviation = st.sidebar.slider("Std Deviation", 0.1, 10.0, 1.0, 0.1)

data = pd.read_csv("./sampledata/Load.csv")

st.title("")

if not data.empty:
    ## s1, data preprocessing
    # c0, create timestamped data
    data['Time'] = pd.to_timedelta(data['Time']).dt.total_seconds()
    dataset = data.to_numpy()
    timestamp = dataset[:, 0]

    # c1, replicate signals
    signals = replicate_signals(dataset, number_of_replicas=100, noise_level=noise_level, noise_seed=0, time_shift_range=time_shift_range)

    # c2, calculate the mean and standard deviation of the replicated signals
    mean = np.mean(signals, axis=0)
    std = np.std(signals, axis=0, ddof=1)

    ## s3, plot all the replicated signals, the original signal, and the mean in the same graph
    fig = go.Figure()
    for i in range(signals.shape[0]): 
        if i == 0:
            fig.add_trace(go.Scatter(x=timestamp, y=signals[i, :], mode='lines', showlegend=True, name='Replicated Signals', line=dict(color='yellow', width=0.5), opacity=.1))
        else:
            fig.add_trace(go.Scatter(x=timestamp, y=signals[i, :], mode='lines', showlegend=False, line=dict(color='yellow', width=0.5), opacity=.1))

    fig.add_trace(go.Scatter(x=timestamp, y=mean, mode='lines', showlegend=True, name="Average over replicates", line=dict(color='#ff0000', width=1), opacity=1.))

    fig.add_trace(go.Scatter(x=timestamp, y=dataset[:, 1], mode='lines', showlegend=True, name="Original signal", line=dict(color='#0000ff', width=2), opacity=1.))

    fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='top'))

    st.plotly_chart(fig)

    ## s4, plot the average, and upper bound and lower bound for 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10 standard deviations from the mean using errorbar plot 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamp, y=mean, mode='lines', showlegend=True, name='Agerage over all replicates', line=dict(color='red', width=1), opacity=.5))
    fig.add_trace(go.Scatter(x=timestamp, y=mean + std_deviation * std, mode='lines', showlegend=True, name=f'CI {std_dev_to_prob(std_deviation):.2f}% Upper Bound', line=dict(color='orange', width=1), opacity=.7))
    fig.add_trace(go.Scatter(x=timestamp, y=mean - std_deviation * std, mode='lines', showlegend=True, name=f'CI {std_dev_to_prob(std_deviation):.2f}% Lower Bound', line=dict(color='orange', width=1), opacity=.7))

    fig.update_layout(legend=dict(x=1, y=1, xanchor='right', yanchor='top'))

    st.plotly_chart(fig)




else:
    st.write("No data to plot.")

