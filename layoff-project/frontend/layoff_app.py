import streamlit as st
from routers import home, about, insights, upload, retrain

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Home", "About", "Insights", "Upload Data", "Retrain Model"]
)

if page == "Home":
    home.show()
elif page == "About":
    about.show()
elif page == "Insights":
    insights.show()
elif page == "Upload Data":
    upload.show()
elif page == "Retrain Model":
    retrain.show()
