import streamlit as st
import requests


def show():
    st.title("Trigger Model Retraining")

    if st.button("Start Retraining"):
        response = requests.post("http://localhost:8000/retrain")
        if response.status_code == 200:
            st.success("Retraining completed!")
            st.text(response.json().get("stdout"))
        else:
            st.error("Failed to trigger retraining.")
