from prophet import Prophet
from prophet.plot import plot_plotly as prophet_plot
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from st_aggrid import AgGrid

st.set_page_config(page_title="Startup Dashboard", layout="wide")

# Hero Section and Styling
st.markdown("""
    <style>
        .stApp { background-color: #121212; color: white; font-family: 'Poppins', sans-serif; }
        h1 { text-align: center; color: #60fc37; font-size: 3rem; font-weight: 900; }
        h4, h2, h3 { color: #ffffff !important; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚀 Startup Ecosystem Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Analyze startup trends using investment patterns, locations, and industries.</h4>", unsafe_allow_html=True)

@st.cache_data
def fetch_startup_data():
    try:
        df = pd.read_csv("startup_funding.csv")
        df.columns = ["Sr No", "Date", "Startup Name", "Industry Vertical", "SubVertical", "City Location", "Investors Name", "Investment Type", "Amount in USD", "Remarks"]
        df["Date"] = pd.to_datetime(df["Date"], format='%d/%m/%Y', errors='coerce')
        df["Amount in USD"] = pd.to_numeric(df["Amount in USD"].str.replace(",", ""), errors='coerce')
        return df.drop(columns=["Sr No", "Remarks"])
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
if "startup_df" not in st.session_state:
    st.session_state.startup_df = fetch_startup_data()

startup_df = st.session_state.startup_df

if not startup_df.empty:
    st.success(f"Loaded {len(startup_df)} records!")

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Startups", len(startup_df))
    col2.metric("Total Funding", f"${startup_df['Amount in USD'].sum():,.0f}")
    col3.metric("Top City", startup_df['City Location'].mode()[0])

    # Filters
    st.markdown("### 🔍 Filter Startups")
    col1, col2, col3 = st.columns(3)
    industries = ["All"] + sorted(startup_df["Industry Vertical"].dropna().unique())
    cities = ["All"] + sorted(startup_df["City Location"].dropna().unique())
    years = ["All"] + sorted(startup_df["Date"].dropna().dt.year.unique())

    selected_industry = col1.selectbox("Industry", industries)
    selected_city = col2.selectbox("City", cities)
    selected_year = col3.selectbox("Year", years)

    filtered_df = startup_df.copy()
    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df["Industry Vertical"] == selected_industry]
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df["City Location"] == selected_city]
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df["Date"].dt.year == selected_year]

    # Show Table
    st.subheader("📋 Filtered Dataset")
    AgGrid(filtered_df)

    # Charts
    st.subheader("💰 Investment by Industry")
    st.plotly_chart(px.bar(filtered_df, x='Industry Vertical', y='Amount in USD', color='Industry Vertical'), use_container_width=True)

    st.subheader("📈 Investment Over Time")
    if not filtered_df['Date'].isnull().all():
        st.plotly_chart(px.line(filtered_df, x='Date', y='Amount in USD', color='Industry Vertical'), use_container_width=True)

    st.subheader("🌆 Top Cities")
    city_funds = filtered_df.groupby("City Location")["Amount in USD"].sum().reset_index()
    st.plotly_chart(px.bar(city_funds.sort_values("Amount in USD", ascending=False)[:10], x="City Location", y="Amount in USD"), use_container_width=True)

    st.subheader("📊 Growth Over Years")
    growth = filtered_df.groupby(filtered_df['Date'].dt.year).size().reset_index(name='Startup Count')
    st.plotly_chart(px.bar(growth, x='Date', y='Startup Count'), use_container_width=True)

    st.subheader("💼 Top Investors")
    investor_counts = filtered_df['Investors Name'].value_counts().head(10).reset_index()
    investor_counts.columns = ['Investor', 'Investments']
    st.plotly_chart(px.bar(investor_counts, x='Investor', y='Investments'), use_container_width=True)

    # Forecasting
    st.subheader("🔮 Forecasting Future Investment Trends")
    if not startup_df["Date"].isnull().all():
        forecast_days = st.slider("Select forecast range (days)", 30, 365, 90, step=30)
        forecast_data = startup_df.groupby("Date")["Amount in USD"].sum().reset_index()
        forecast_data = forecast_data.rename(columns={"Date": "ds", "Amount in USD": "y"})

        model = Prophet()
        model.fit(forecast_data)

        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        forecast = model.predict(future)

        delta = ((forecast['yhat'].iloc[-1] - forecast_data['y'].iloc[-1]) / forecast_data['y'].iloc[-1]) * 100
        direction = "increase" if delta > 0 else "decrease"
        st.markdown(f"🧠 **Forecast:** Funding is expected to **{direction}** by **{abs(delta):.2f}%** in the next **{forecast_days} days**.")

        st.plotly_chart(prophet_plot(model, forecast), use_container_width=True)

        # Download
        forecast_csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False).encode()
        st.download_button("📥 Download Forecast Data", data=forecast_csv,
