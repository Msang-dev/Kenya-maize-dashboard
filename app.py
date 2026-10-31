import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Kenya 47-County Maize Intelligence Hub",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

#LOAD UPLOADED EXCEL DATASET
@st.cache_data
def load_maize_data():
    file_path = 'Maize-Production-2012-2020-Combined.xlsx'
    df_s4 = pd.read_excel(file_path, sheet_name='Sheet4')
    
    # Clean county names (strip trailing spaces)
    df_s4['County'] = df_s4['County'].astype(str).str.strip()
    
    # Pivot to get clean wide tables per indicator
    df_pivot = df_s4.pivot_table(index='County', columns=['Year', 'Indicator'], values='Value')
    
    # Extract Production, Area, and Yield for historical analysis
    prod_pivot = df_s4[df_s4['Indicator'] == 'Production (MT)'].pivot(index='County', columns='Year', values='Value')
    area_pivot = df_s4[df_s4['Indicator'] == 'Area (HA)'].pivot(index='County', columns='Year', values='Value')
    yield_pivot = df_s4[df_s4['Indicator'] == 'Yield(MT/HA)'].pivot(index='County', columns='Year', values='Value')
    
    return df_s4, prod_pivot, area_pivot, yield_pivot

df_s4, prod_pivot, area_pivot, yield_pivot = load_maize_data()

#SIDEBAR CONTROLS
st.sidebar.image("https://images.unsplash.com/photo-1599814529177-3e445037eb89?auto=format&fit=crop&w=400&q=80", use_container_width=True)
st.sidebar.title("🇰🇪 Kenya Maize AI Suite")
st.sidebar.markdown("Empirical Data & Forecasting Engine")

counties_list = sorted(prod_pivot.index.tolist())
selected_county = st.sidebar.selectbox("Select County (47 Devolved Units)", counties_list)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Simulation Parameters")
sim_rainfall = st.sidebar.slider("Seasonal Rainfall Adjustment (mm)", 300, 1500, 850)
sim_fertilizer = st.sidebar.slider("Fertilizer Application Intensity (kg/acre)", 0, 100, 40)
seed_variety = st.sidebar.selectbox("Certified Seed Hybrid", ["H614 (Highland Hybrid)", "H6213 (Drought Tolerant)", "DK777 (Commercial Ultra)", "Local Open Pollinated"])

#PREDICTIVE FORECASTING MODEL (2012-2020 TRENDS)
def get_county_forecast(county):
    row_prod = prod_pivot.loc[county].dropna()
    years = row_prod.index.values.astype(float)
    production = row_prod.values.astype(float)
    
    if len(years) >= 3:
        m, c = np.polyfit(years, production, 1)
        pred_2026 = max(0, m * 2026 + c)
        pred_2030 = max(0, m * 2030 + c)
    else:
        pred_2026 = production[-1] if len(production) > 0 else 0
        pred_2030 = pred_2026
        
    latest_prod = production[-1] if len(production) > 0 else 0
    latest_area = area_pivot.loc[county].dropna().iloc[-1] if county in area_pivot.index and len(area_pivot.loc[county].dropna()) > 0 else 1000
    latest_yield = yield_pivot.loc[county].dropna().iloc[-1] if county in yield_pivot.index and len(yield_pivot.loc[county].dropna()) > 0 else 2.0
    
    return latest_prod, latest_area, latest_yield, pred_2026, pred_2030, row_prod

latest_prod, latest_area, latest_yield, pred_2026, pred_2030, county_history = get_county_forecast(selected_county)

# Adjust forecast with simulation sliders
seed_multiplier = 1.25 if "Drought" in seed_variety and sim_rainfall < 600 else (1.15 if "DK777" in seed_variety else 1.0)
fert_multiplier = 1.0 + (sim_fertilizer / 200.0)
adjusted_2026_prod = int(pred_2026 * seed_multiplier * fert_multiplier)

#MAIN DASHBOARD INTERFACE
st.title(f"🌽 Kenya Maize Intelligence Dashboard: {selected_county}")
st.markdown(f"**Devolved Unit Analysis:** Historical Production Records & Predictive Trend Modeling (2012–2030)")

# KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Latest Recorded Prod. (2020)", value=f"{int(latest_prod):,} MT", delta="Baseline Ministry Data")
with col2:
    st.metric(label="Harvested Area", value=f"{int(latest_area):,} HA", delta=f"Yield: {round(latest_yield, 2)} MT/HA")
with col3:
    st.metric(label="Projected Output (2026)", value=f"{adjusted_2026_prod:,} MT", delta=f"{round(((adjusted_2026_prod - latest_prod)/max(1, latest_prod))*100, 1)}% Growth vs 2020")
with col4:
    st.metric(label="Long-Term Outlook (2030)", value=f"{int(pred_2030):,} MT", delta="Linear Trend Forecast")

st.markdown("---")

# Tabs for Multi-Perspective Analytics
tab1, tab2, tab3, tab4 = st.tabs(["📈 Historical & Trend Forecast", "📊 National County Ranking", "🗺️ Area vs Production Analysis", "💡 Agronomic Insights"])

with tab1:
    st.subheader(f"Historical Production & 2030 Trajectory for {selected_county}")
    hist_df = pd.DataFrame({'Year': county_history.index, 'Production (MT)': county_history.values})
    
    fig_line = px.line(
        hist_df, x='Year', y='Production (MT)', markers=True,
        title=f"Historical Maize Production (2012-2020) with Linear Regression Forecast",
        color_discrete_sequence=["#1b4332"]
    )
    # Add forecast points
    fig_line.add_scatter(x=[2026, 2030], y=[pred_2026, pred_2030], mode='markers+text', text=[f"2026: {int(pred_2026):,} MT", f"2030: {int(pred_2030):,} MT"], textposition="top center", name="Forecast")
    fig_line.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    st.subheader("Top Maize Producing Counties (Based on 2020 Dataset)")
    top_2020 = prod_pivot[2020].sort_values(ascending=False).head(15).reset_index()
    top_2020.columns = ['County', 'Production_2020_MT']
    
    fig_bar = px.bar(
        top_2020, x='County', y='Production_2020_MT', text='Production_2020_MT',
        title="Top 15 Producing Counties in Metric Tons (2020)",
        color='Production_2020_MT', color_continuous_scale="Greens"
    )
    fig_bar.update_layout(template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.subheader("Harvested Area (HA) vs Production (MT) Distribution")
    combined_scatter_df = pd.DataFrame({
        'County': prod_pivot.index,
        'Production_2020': prod_pivot[2020].values,
        'Area_2020': area_pivot[2020].values if 2020 in area_pivot.columns else area_pivot.iloc[:,-1].values
    }).dropna()
    
    fig_scatter = px.scatter(
        combined_scatter_df, x='Area_2020', y='Production_2020', text='County',
        size='Production_2020', hover_name='County',
        title="County Land Utilization vs. Harvest Output (2020)",
        labels={'Area_2020': 'Harvested Area (Hectares)', 'Production_2020': 'Production (Metric Tons)'}
    )
    fig_scatter.update_traces(textposition='top center')
    fig_scatter.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab4:
    st.subheader(f"🤖 Strategic Recommendations for {selected_county}")
    st.markdown(f"""
    * **Production Trajectory:** Based on historical data from 2012 to 2020, **{selected_county}** demonstrates a baseline production capacity of **{int(latest_prod):,} MT** across **{int(latest_area):,} hectares**.
    * **Fertilizer & Input Planning:** With your simulated fertilizer adoption intensity at **{sim_fertilizer} kg/acre**, extension officers should prioritize subsidized DAP and CAN fertilizer distribution to bridge nutrient gaps.
    * **Food Security Reserves:** National Cereals and Produce Board (NCPB) depots in the region should prepare storage infrastructure to handle the projected 2026 harvest target of **{adjusted_2026_prod:,} MT**.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d;'>Kenya 47-County Maize Intelligence Engine | Powered by Official Ministry of Agriculture Datasets</p>", unsafe_allow_html=True)
   
