import streamlit as st
from routers import home, about, insights, upload, metrics, ats_system

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "About",
        "Insights",
        "Metrics",
        "Upload and Retrain Data",
        "ATS_System",
    ],
)

if page == "Home":
    home.show()
elif page == "About":
    about.show()
elif page == "Insights":
    insights.show()
elif page == "Upload and Retrain Data":
    upload.show()
elif page == "Metrics":
    metrics.show()
elif page == "ATS_System":
    ats_system.show()
