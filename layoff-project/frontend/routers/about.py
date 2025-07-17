import streamlit as st


def show():
    st.title("About This Project")
    st.markdown("""
  Layoff Insight is a machine learning web app that predicts the severity of corporate layoffs. Given company details such as number of employees laid off, percentage of workforce affected, industry, and funding stage, it classifies layoffs into four severity levels: Low, Medium, High, or Unknown.

The project includes:

    A trained ML pipeline (XGBoost model)

    A FastAPI backend for real-time predictions

    A Streamlit frontend for user-friendly inputs and results

This solution helps analysts, HR teams, and researchers quickly assess the potential impact of layoffs across industries.
    Github: https://github.com/YanivBohbot/ML_PROJECT_Layoff_Insight
    """)
