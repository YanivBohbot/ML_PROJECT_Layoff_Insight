import streamlit as st
import pandas as pd
import plotly.express as px


import requests

API_URL = "http://localhost:8000/data"


def show():
    st.title("📊 Layoff Data Insights Dashboard")

    response = requests.get(API_URL)
    if response.status_code != 200:
        st.error("Failed to fetch data from backend.")
        return

    # Convert JSON to DataFrame
    data = response.json()  # parses the JSON string
    df = pd.DataFrame(data)

    # Extract year from date
    if "year" not in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

    # Define a mapping function for industry_type

    def map_industry_type(industry):
        if pd.isna(industry):
            return "Unknown"
        industry = industry.lower()
        if "travel" in industry or "hospitality" in industry:
            return "Travel"
        elif "health" in industry:
            return "Healthcare"
        elif "security" in industry:
            return "Security"
        elif "marketing" in industry or "media" in industry:
            return "Marketing"
        elif "data" in industry or "analytics" in industry:
            return "Data"
        elif "logistic" in industry or "transport" in industry:
            return "Logistics"
        elif "sales" in industry or "retail" in industry:
            return "Sales"
        elif "ai" in industry or "artificial" in industry:
            return "AI"
        else:
            return "Other"

    # Apply mapping to create industry_type column
    df["industry_type"] = df["industry"].apply(map_industry_type)

    # Convert percentage_laid_off to float (safely)
    df["percentage_laid_off"] = pd.to_numeric(
        df["percentage_laid_off"], errors="coerce"
    )

    # Define the severity mapping
    def map_severity(p):
        if pd.isna(p):
            return "Unknown"
        elif p < 0.3:
            return "Low"
        elif p < 0.7:
            return "Medium"
        else:
            return "High"

    # Apply it
    df["severity_label"] = df["percentage_laid_off"].apply(map_severity)

    # Sidebar filter
    st.sidebar.header("📌 Dashboard Navigation")
    options = st.sidebar.multiselect(
        "Choose charts to display:",
        [
            "Layoffs by Industry Type Over Years",
            "Number of Layoffs Per Year",
            "Layoff Severity Distribution by Country",
            "Layoff Severity Distribution by Industry",
            "Distribution of Layoff Percentages",
            "Top 10 Countries by Number of Layoffs",
            "Top 10 Industries by Number of Layoffs",
            "Yearly Layoffs Trend",
        ],
        default=[
            "Layoffs by Industry Type Over Years",
            "Number of Layoffs Per Year",
            "Yearly Layoffs Trend",
            "Top 10 Countries by Number of Layoffs",
        ],
    )

    # 1. Layoffs by Industry Type Over Years
    if "Layoffs by Industry Type Over Years" in options:
        industry_year = (
            df.groupby(["year", "industry_type"])["total_laid_off"].sum().reset_index()
        )
        fig1 = px.bar(
            industry_year,
            x="year",
            y="total_laid_off",
            color="industry_type",
            title="Layoffs by Industry Type Over the Years",
            barmode="group",
        )
        st.plotly_chart(fig1)

    # 2. Number of Layoffs Per Year
    if "Number of Layoffs Per Year" in options:
        yearly = df.groupby("year")["total_laid_off"].sum().reset_index()
        fig2 = px.bar(
            yearly, x="year", y="total_laid_off", title="Total Layoffs Per Year"
        )
        st.plotly_chart(fig2)

    # 3. Layoff Severity Distribution by Country
    if "Layoff Severity Distribution by Country" in options:
        fig3 = px.histogram(
            df,
            x="country",
            color="severity_label",
            barmode="stack",
            title="Layoff Severity by Country",
        )
        st.plotly_chart(fig3)

    # 4. Layoff Severity Distribution by Industry
    if "Layoff Severity Distribution by Industry" in options:
        fig4 = px.histogram(
            df,
            x="industry_type",
            color="severity_label",
            barmode="stack",
            title="Layoff Severity by Industry Type",
        )
        st.plotly_chart(fig4)

    # 5. Distribution of Layoff Percentages
    if "Distribution of Layoff Percentages" in options:
        fig5 = px.histogram(
            df,
            x="percentage_laid_off",
            nbins=30,
            title="Distribution of Layoff Percentages",
            labels={"percentage_laid_off": "Percentage Laid Off"},
        )
        st.plotly_chart(fig5)
        st.write("🔎 Preview of percentage_laid_off values:")
        st.write(df["percentage_laid_off"].describe())
        st.write(df["percentage_laid_off"].head(10))

    # 6. "Top 10 Countries by Number of Layoffs"
    if "Top 10 Countries by Number of Layoffs" in options:
        st.subheader("Top 10 Countries by Number of Layoffs")
        country_counts = (
            df.groupby("country")["total_laid_off"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(country_counts)

    # 7. "Top 10 Industries by Number of Layoffs"
    if "Top 10 Industries by Number of Layoffs" in options:
        st.subheader("Top 10 Industries by Number of Layoffs")
        industry_counts = (
            df.groupby("industry")["total_laid_off"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(industry_counts)

    # 8. "Yearly Layoffs Trend"
    if "Yearly Layoffs Trend" in options:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce", infer_datetime_format=True
        )

        # Remove rows with invalid or missing dates
        df = df.dropna(subset=["date"])

        # Ensure total_laid_off is numeric
        df["total_laid_off"] = pd.to_numeric(
            df["total_laid_off"], errors="coerce"
        ).fillna(0)

        # Extract year as integer
        df["year"] = df["date"].dt.year.astype(int)

        # ---------------------
        # Aggregate yearly
        # ---------------------
        yearly_trend = (
            df.groupby("year", as_index=False)["total_laid_off"]
            .sum()
            .sort_values("year")
        )
        yearly_trend["year"] = yearly_trend["year"].astype(int)
        # Display chart with proper x-axis
        st.subheader("Yearly Layoffs Trend")
        st.line_chart(yearly_trend.set_index("year"))  # x-axis
