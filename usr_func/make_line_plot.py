import streamlit as st
import pandas as pd


def make_line_plot(data, color="#FF5733"):
    """ Use st.line_chart to make the line plot, as st.line_chart only accepts one dataframe, we need to wrap the numpy array to a dataframe before plotting, and appending the column names to the dataframe. 
    """
    data_df = pd.DataFrame(data, columns=["Time", "Value"])
    st.line_chart(data_df, x='Time', y='Value', color=color)