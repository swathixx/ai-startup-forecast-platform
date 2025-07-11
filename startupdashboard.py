from prophet import Prophet
from prophet.plot import plot_plotly as prophet_plot

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from st_aggrid import AgGrid

os.environ['PYTHONUTF8'] = '1'

st.set_page_config(page_title="Startup  Dashboard", layout="wide")

# Hero and styling
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        .stApp {
            background: #121212;
        }
        h1 {
            text-align: center;
            color: #60fc37 !important;
            font-size: 3rem;
            font-weight: 900;
        }
        h2, h3, h4 {
            color: #ffffff !important;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1>🚀 Startup Ecosystem Dashboard</h1>
    <h4 style='text-align: center; color: #ffffff;'>Analyze startup trends using investment patterns, locations, and industries.</h4>
""", unsafe_allow_html=True)

@st.cache_data
def fetch_startup_data():
    try:
        df = pd.read_csv("startup_funding.csv")
        df.columns = ["Sr No", "Date", "Startup Name", "Industry Vertical", "SubVertical", "City Location", "Investors Name", "Investment Type", "Amount in USD", "Remarks"]
        df["Date"] = pd.to_datetime(df["Date"], format='%d/%m/%Y', errors='coerce')
        df["Amount in USD"] = pd.to_numeric(df["Amount in USD"].str.replace(",", ""), errors='coerce')
        return df.drop(columns=["Sr No", "Remarks"])
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

if "startup_df" not in st.session_state:
    st.session_state.startup_df = fetch_startup_data()

startup_df = st.session_state.startup_df

if not startup_df.empty:
    st.success(f"Loaded {len(startup_df)} startups!")

    # KPI Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Startups", len(startup_df))
    col2.metric("Total Funding", f"${startup_df['Amount in USD'].sum():,.0f}")
    col3.metric("Top City", startup_df['City Location'].mode()[0])

    # Filters
    st.markdown("### 🔍 Filter Startups")
    col1, col2, col3 = st.columns(3)
    industries = ["All"] + sorted(startup_df["Industry Vertical"].dropna().unique().tolist())
    cities = ["All"] + sorted(startup_df["City Location"].dropna().unique().tolist())
    years = ["All"] + sorted(startup_df["Date"].dropna().dt.year.unique().tolist())

    selected_industry = col1.selectbox("Select Industry", industries)
    selected_city = col2.selectbox("Select City", cities)
    selected_year = col3.selectbox("Select Year", years)

    filtered_df = startup_df.copy()
    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df["Industry Vertical"] == selected_industry]
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df["City Location"] == selected_city]
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df["Date"].dt.year == selected_year]

    # Show data
    st.write("### 📋 Filtered Startup Dataset")
    AgGrid(filtered_df)

    st.subheader("💰 Investment Distribution by Industry")
    fig_industry = px.bar(filtered_df, x='Industry Vertical', y='Amount in USD', color='Industry Vertical', height=500)
    st.plotly_chart(fig_industry, use_container_width=True)

    st.subheader("📈 Investment Over Time")
    if not filtered_df['Date'].isnull().all():
        fig_time = px.line(filtered_df, x='Date', y='Amount in USD', color='Industry Vertical', markers=True)
        st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("🌆 Top Cities by Investment")
    city_investment = filtered_df.groupby("City Location")["Amount in USD"].sum().reset_index()
    fig_city = px.bar(city_investment.sort_values(by='Amount in USD', ascending=False)[:10], 
                      x='City Location', y='Amount in USD', color='City Location')
    st.plotly_chart(fig_city, use_container_width=True)

    st.subheader("📊 Growth of Startups Over the Years")
    startup_counts = filtered_df.groupby(filtered_df['Date'].dt.year).size().reset_index(name='Startup Count')
    fig_growth = px.bar(startup_counts, x='Date', y='Startup Count', color='Startup Count')
    st.plotly_chart(fig_growth, use_container_width=True)

    st.subheader("💼 Top Investors")
    investor_counts = filtered_df['Investors Name'].value_counts().reset_index()
    investor_counts.columns = ['Investor', 'Number of Investments']
    fig_investors = px.bar(investor_counts.head(10), x='Investor', y='Number of Investments', color='Number of Investments')
    st.plotly_chart(fig_investors, use_container_width=True)

    # Forecast Section
    st.subheader("📊 Market Prediction: Future Funding Trends")
    if not startup_df["Date"].isnull().all():
        st.markdown("#### 📅 Select Forecast Period (in Days)")
        forecast_days = st.slider("Forecast Period", 30, 365, 90, step=30)

        funding_data = startup_df.groupby("Date")["Amount in USD"].sum().reset_index()
        funding_data = funding_data.rename(columns={"Date": "ds", "Amount in USD": "y"})

        model = Prophet()
        model.fit(funding_data)

        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        forecast = model.predict(future)

        last_known = funding_data["y"].iloc[-1]
        predicted = forecast["yhat"].iloc[-1]
        delta = ((predicted - last_known) / last_known) * 100

        st.markdown(f"🧠 **Forecast Summary**: Funding is expected to {'increase' if delta > 0 else 'decrease'} by **{abs(delta):.2f}%** over the next **{forecast_days} days**.")

        fig_forecast = prophet_plot(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)

        csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast_data.csv", mime="text/csv")
    else:
        st.warning("Not enough data for market prediction.")
else:
    st.warning("No startup data found.")

st.write("<p style='text-align: center;'>Made with ❤️ using Streamlit, Pandas, and Plotly</p>", unsafe_allow_html=True)