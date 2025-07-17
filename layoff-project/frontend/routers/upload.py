import streamlit as st
import pandas as pd


def show():
    st.title("Upload New Training Data")

    uploaded_file = st.file_uploader("Upload CSV file for retraining", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head())
