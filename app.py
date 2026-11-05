import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Kenya 47-County Maize Intelligence Hub",
    page_icon="🇰🇪",
    layout="wide"
)

def load_data_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls', '.xlsm')):
            xls = pd.ExcelFile(uploaded_file)
            if 'Sheet4' in xls.sheet_names:
                return pd.read_excel(uploaded_file, sheet_name='Sheet4')
            else:
                return pd.read_excel(uploaded_file, sheet_name=xls.sheet_names[0])
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return pd.DataFrame()

st.sidebar.title("🇰🇪 Kenya Maize AI Pred")
st.sidebar.markdown("Empirical Data & Forecasting Engine")

uploaded_file = st.sidebar.file_uploader("Upload Maize Production Dataset (Excel/CSV/XLSM)", type=["xlsx", "xls", "xlsm", "csv"])

if uploaded_file is not None:
    df_data = load_data_file(uploaded_file)
else:
    try:
        df_data = pd.read_excel("Maize-Production-2012-2020-Combined.xlsx", sheet_name="Sheet4")
    except Exception:
        try:
            df_data = pd.read_csv("maize_production.csv")
        except Exception:
            df_data = pd.DataFrame(columns=["County", "Year", "Area (HA)", "Production (MT)", "Yield(MT/HA)"])
            st.sidebar.warning("Please upload your .xlsm dataset using the file uploader above.")

if not df_data.empty and "County" in df_data.columns:
    counties_list = sorted(df_data["County"].unique().tolist())
    selected_county = st.sidebar.selectbox(
        "Select County (47 Devolved Units)",
        counties_list,
        index=counties_list.index("Trans Nzoia") if "Trans Nzoia" in counties_list else 0
    )
else:
    selected_county = st.sidebar.selectbox("Select County (47 Devolved Units)", ["Trans Nzoia"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Simulation Parameters")
rainfall = st.sidebar.slider("Seasonal Rainfall Adjustment (mm)", 300, 1500, 850)
fertilizer = st.sidebar.slider("Fertilizer Application Intensity (kg/acre)", 0, 100, 40)
seed_type = st.sidebar.selectbox("Certified Seed Hybrid", ["H614 (Highland Hybrid)", "DH04 (Dryland Hybrid)", "DK777 (Commercial Ultra)", "Local Open-Pollinated"])

if not df_data.empty and "County" in df_data.columns:
    county_df = df_data[df_data["County"] == selected_county].sort_values("Year")
    latest_row = county_df[county_df["Year"] == 2020]
    if not latest_row.empty:
        latest_prod = latest_row["Production (MT)"].values[0]
        harvested_area = latest_row["Area (HA)"].values[0]
        yield_val = latest_row["Yield(MT/HA)"].values[0]
    else:
        latest_prod = county_df["Production (MT)"].iloc[-1] if not county_df.empty else 0
        harvested_area = county_df["Area (HA)"].iloc[-1] if not county_df.empty else 0
        yield_val = county_df["Yield(MT/HA)"].iloc[-1] if not county_df.empty else 0
else:
    county_df = pd.DataFrame()
    latest_prod, harvested_area, yield_val = 130151, 68686, 1.89

simulated_prod = int(latest_prod * (rainfall / 850) * (1 + (fertilizer / 200)) * (1.1 if "Hybrid" in seed_type else 0.95))
pct_change = ((simulated_prod - latest_prod) / latest_prod) * 100 if latest_prod > 0 else 0

st.title(f"🌽 Kenya Maize Intelligence Dashboard: {selected_county}")
st.markdown(f"*Devolved Unit Analysis:* Historical Production Records & Predictive Trend Modelling (2012–2030)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Latest Recorded Production (2020)", value=f"{latest_prod:,.0f} MT")
with col2:
    st.metric(label="Harvested Area", value=f"{harvested_area:,.0f} HA")
with col3:
    st.metric(label="Yield", value=f"{yield_val:,.2f} MT/HA")
with col4:
    st.metric(label="Projected Output (2026)", value=f"{simulated_prod:,.0f} MT", delta=f"{pct_change:.1f}% vs latest recorded year")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📈 Historical & Forecast", "📊 County Ranking", "💡 Agronomic Insights"])

with tab1:
    st.subheader(f"Production Trajectory for {selected_county} (2012–2030)")
    if not county_df.empty:
        hist_years = county_df["Year"].tolist()
        hist_prod = county_df["Production (MT)"].tolist()
    else:
        hist_years, hist_prod = [2020], [130151]
        
    future_years = list(range(2021, 2031))
    last_val = hist_prod[-1] if hist_prod else 100000
    future_prod = [int(last_val * (1 + 0.02 * (y - 2020) * (rainfall / 850))) for y in future_years]
    
    full_years = hist_years + future_years
    full_prod = hist_prod + future_prod
    
    trend_df = pd.DataFrame({"Year": full_years, "Production (MT)": full_prod})
    
    fig = px.line(trend_df, x="Year", y="Production (MT)", markers=True, title=f"Annual Maize Yield & Projections: {selected_county}")
    fig.add_vrect(x0=2020, x1=2030, fillcolor="green", opacity=0.1, annotation_text="Forecast Window", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Top 15 Maize Producing Counties (2020)")
    if not df_data.empty and "Year" in df_data.columns:
        df_2020 = df_data[df_data["Year"] == 2020].sort_values(by="Production (MT)", ascending=False)
        top_15 = df_2020.head(15)
        fig_bar = px.bar(top_15, x="County", y="Production (MT)", text="Production (MT)", title="Top Producing Counties in 2020")
        fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(top_15[["County", "Production (MT)", "Area (HA)", "Yield(MT/HA)"]].rename(columns={"Production (MT)": "Production (MT)", "Area (HA)": "Area (HA)", "Yield(MT/HA)" : "Yield (MT/HA)"}), use_container_width=True)
    else:
        st.info("Please upload a valid dataset to view county rankings.")

with tab3:
    st.subheader("Agronomic Recommendations & Climate Adaptation")
    st.info(f"Analysis for **{selected_county}** based on rainfall input of **{rainfall} mm** and fertilizer input of **{fertilizer} kg/acre**.")
    st.write("- **Soil Health Management:** Rotate maize with legumes to replenish nitrogen levels naturally.")
    st.write("- **Water Conservation:** Implement micro-catchment tillage if seasonal rainfall falls below 600 mm.")
    st.write("- **Pest Control:** Monitor closely for Fall Armyworm outbreaks during early vegetative growth stages.")

st.markdown("---")
st.markdown("Kenya 47-County Maize Intelligence Engine | Powered by Python, Streamlit, Pandas, NumPy and Plotly")
