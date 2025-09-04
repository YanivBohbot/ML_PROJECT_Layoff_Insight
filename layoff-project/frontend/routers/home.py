import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL")


def show():
    st.title("Layoff Severity Prediction App")
    st.write("""
        **Field Explanations:**

        - **Total Laid Off:** Total number of employees laid off in this event.
        - **Percentage Laid Off:** Fraction of the company’s workforce let go (0–100%).
        - **Funds Raised:** How much money the company has raised from investors.
        - **Industry:** Sector where the company operates (e.g. AI, Crypto).
        - **Country:** Country where the layoff took place.
        - **Stage:** Funding stage of the company (e.g. Seed, Series A).
    """)

    industries = ["AI", "Crypto", "Finance", "Retail", "Logistics", "Travel", "Other"]

    countries = ["United States", "India", "Nigeria", "Germany", "Canada"]

    stages = [
        "Seed",
        "Series A",
        "Series B",
        "Series C",
        "Post-IPO",
        "Acquired",
        "Unknown",
    ]

    total_laid_off = st.number_input("Total Laid Off in 2020-2024", min_value=0)
    perc_laid_off = st.number_input(
        "Percentage Laid Off of workforce in companie", min_value=0.0, max_value=100.0
    )
    funds_raised = st.text_input("Funds Raised (e.g. $120K)")
    industry = st.selectbox("Industry", industries)
    country = st.selectbox("Country", countries)
    stage = st.selectbox("Stage", stages)

    if st.button("Predict Severity"):
        payload = {
            "total_laid_off": total_laid_off,
            "perc_laid_off": perc_laid_off,
            "funds_raised": funds_raised,
            "industry": industry,
            "country": country,
            "stage": stage,
        }

        response = requests.post(f"{API_BASE_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Severity: {result['severity_label']}")
        else:
            st.error(f"Error {response.status_code}: Unable to get prediction.")
