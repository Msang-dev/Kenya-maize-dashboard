from functools import wraps
import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Kenya 47-County Maize Intelligence Hub",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main {
    background-color: #f4f6f9;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
""",
    unsafe_allow_html=True,
)


def execution_timer(func):

  @wraps(func)
  def wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    # Use st.toast or remove print for cloud environments
    return result

  return wrapper


@st.cache_data
@execution_timer
def load_maize_data():
  file_path = "Maize-Production-2012-2020-Combined.xlsx"

  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset '{file_path}' not found.")

  df = pd.read_excel(file_path, sheet_name="Sheet4")

  required_columns = ["County", "Year", "Indicator", "Value"]
  missing_columns = [col for col in required_columns if col not in df.columns]

  if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

  df["County"] = df["County"].astype(str).str.strip()

  # Clean indicator names
  df["Indicator"] = df["Indicator"].astype(str).str.strip()

  # Convert numerical columns
  df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
  df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

  # Remove incomplete records
  df = df.dropna(subset=["County", "Year", "Indicator", "Value"])

  df["Year"] = df["Year"].astype(int)

  # Remove invalid county names
  df = df[df["County"].str.lower() != "nan"]

  # Create pivot tables
  prod_pivot = (
      df[df["Indicator"] == "Production (MT)"]
      .pivot_table(index="County", columns="Year", values="Value", aggfunc="mean")
  )

  area_pivot = (
      df[df["Indicator"] == "Area (HA)"]
      .pivot_table(index="County", columns="Year", values="Value", aggfunc="mean")
  )

  yield_pivot = (
      df[df["Indicator"] == "Yield(MT/HA)"]
      .pivot_table(index="County", columns="Year", values="Value", aggfunc="mean")
  )

  return df, prod_pivot, area_pivot, yield_pivot


def county_generator(production_table, year):
  if year not in production_table.columns:
    return

  for county, production in production_table[year].items():
    if pd.notna(production):
      yield {"County": county, "Production (MT)": float(production)}


def county_iterator(counties):
  county_iter = iter(counties)
  first_county = next(county_iter, "No county available")
  return first_county


def recursive_total(values, index=0):
  # Base condition: stop when all values are processed.
  if index >= len(values):
    return 0.0

  current_value = float(values[index])
  return current_value + recursive_total(values, index + 1)


def get_county_forecast(county, prod_pivot, area_pivot, yield_pivot):
  # Extract historical production
  if county not in prod_pivot.index:
    return 0, 0, 0, 0, 0, pd.Series(dtype=float)

  row_prod = prod_pivot.loc[county].dropna().sort_index()

  years = row_prod.index.to_numpy(dtype=float)
  production = row_prod.to_numpy(dtype=float)

  # Check whether enough records exist
  if len(years) >= 3:
    m, c = np.polyfit(years, production, 1)
    pred_2026 = max(0, m * 2026 + c)
    pred_2030 = max(0, m * 2030 + c)
  elif len(years) > 0:
    pred_2026 = production[-1]
    pred_2030 = production[-1]
  else:
    pred_2026 = 0
    pred_2030 = 0

  latest_prod = production[-1] if len(production) > 0 else 0

  # Latest harvested area
  if county in area_pivot.index:
    area_values = area_pivot.loc[county].dropna().sort_index()
    latest_area = area_values.iloc[-1] if len(area_values) > 0 else 0
  else:
    latest_area = 0

  # Latest maize yield
  if county in yield_pivot.index:
    yield_values = yield_pivot.loc[county].dropna().sort_index()
    latest_yield = yield_values.iloc[-1] if len(yield_values) > 0 else 0
  else:
    latest_yield = 0

  return (
      latest_prod,
      latest_area,
      latest_yield,
      pred_2026,
      pred_2030,
      row_prod,
  )


try:
  df, prod_pivot, area_pivot, yield_pivot = load_maize_data()

  if prod_pivot.empty:
    st.error("No maize production records were found.")
    st.stop()

except FileNotFoundError:
  st.error(
      "Excel dataset not found. Please ensure "
      "`Maize-Production-2012-2020-Combined.xlsx` is uploaded to your "
      "repository root folder."
  )
  st.stop()

except Exception as error:
  st.error(f"Error loading dataset: {error}")
  st.stop()

st.sidebar.image(
    "https://images.unsplash.com/photo-1599814529177-3e445037eb89?auto=format&fit=crop&w=400&q=80",
    use_column_width=True,
)

st.sidebar.title("🇰🇪 Kenya Maize AI Pred")
st.sidebar.markdown("Empirical Data & Forecasting Engine")

counties_list = sorted(prod_pivot.index.tolist())
first_county = county_iterator(counties_list)

st.sidebar.caption(f"First county from iterator: {first_county}")

selected_county = st.sidebar.selectbox(
    "Select County (47 Devolved Units)", counties_list
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Simulation Parameters")

sim_rainfall = st.sidebar.slider("Seasonal Rainfall Adjustment (mm)", 300, 1500, 850)
sim_fertilizer = st.sidebar.slider(
    "Fertilizer Application Intensity (kg/acre)", 0, 100, 40
)

seed_variety = st.sidebar.selectbox(
    "Certified Seed Hybrid",
    [
        "H614 (Highland Hybrid)",
        "H6213 (Drought Tolerant)",
        "DK777 (Commercial Ultra)",
        "Local Open Pollinated",
    ],
)

(
    latest_prod,
    latest_area,
    latest_yield,
    pred_2026,
    pred_2030,
    county_history,
) = get_county_forecast(selected_county, prod_pivot, area_pivot, yield_pivot)

# Conditional statements for seed selection
if "Drought" in seed_variety and sim_rainfall < 600:
  seed_multiplier = 1.25
elif "DK777" in seed_variety:
  seed_multiplier = 1.15
else:
  seed_multiplier = 1.0

# Fertilizer adjustment
fert_multiplier = 1.0 + sim_fertilizer / 200.0

# Adjusted production forecast
adjusted_2026_prod = int(pred_2026 * seed_multiplier * fert_multiplier)

latest_year = int(max(prod_pivot.columns))
national_values = prod_pivot[latest_year].dropna().tolist()

# Recursive function calculates national production
national_total = recursive_total(national_values)

# MAIN DASHBOARD
st.title(f"🌽 Kenya Maize Intelligence Dashboard: {selected_county}")
st.markdown(
    "**Devolved Unit Analysis:** Historical Production Records & Predictive"
    " Trend Modelling (2012–2030)"
)
st.caption(f"Latest available dataset year: {latest_year}")

col1, col2, col3, col4 = st.columns(4)

with col1:
  st.metric(
      label=f"Latest Recorded Production ({latest_year})",
      value=f"{int(latest_prod):,} MT",
  )

with col2:
  st.metric(
      label="Harvested Area",
      value=f"{int(latest_area):,} HA",
      delta=f"Yield: {latest_yield:.2f} MT/HA",
  )

with col3:
  growth = ((adjusted_2026_prod - latest_prod) / max(1, latest_prod)) * 100
  st.metric(
      label="Projected Output (2026)",
      value=f"{adjusted_2026_prod:,} MT",
      delta=f"{growth:.1f}% vs latest recorded year",
  )

with col4:
  st.metric(
      label=f"National Production ({latest_year})",
      value=f"{int(national_total):,} MT",
  )

st.markdown("---")

# DASHBOARD TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Historical & Forecast",
        "📊 County Ranking",
        "🗺️ Area vs Production",
        "💡 Agronomic Insights",
        "🐍 Python Concepts",
    ]
)

# TAB 1: HISTORICAL PRODUCTION
with tab1:
  st.subheader(f"Historical Production & Forecast for {selected_county}")

  hist_df = pd.DataFrame(
      {"Year": county_history.index, "Production (MT)": county_history.values}
  )

  fig_line = px.line(
      hist_df,
      x="Year",
      y="Production (MT)",
      markers=True,
      title="Historical Maize Production",
  )

  fig_line.add_trace(
      go.Scatter(
          x=[2026, 2030],
          y=[pred_2026, pred_2030],
          mode="markers+text",
          text=[
              f"2026: {int(pred_2026):,} MT",
              f"2030: {int(pred_2030):,} MT",
          ],
          textposition="top center",
          name="Linear Forecast",
      )
  )

  fig_line.update_layout(
      template="plotly_white",
      height=500,
      xaxis_title="Year",
      yaxis_title="Production (Metric Tons)",
  )

  st.plotly_chart(fig_line, use_container_width=True)

  st.info(
      "Forecasts are based on historical linear trends. They are estimates, not"
      " guaranteed harvest outcomes."
  )

# TAB 2: COUNTY RANKING
with tab2:
  st.subheader(f"Top 15 Maize Producing Counties ({latest_year})")

  generated_records = list(county_generator(prod_pivot, latest_year))
  ranking_df = pd.DataFrame(generated_records)

  top_15 = (
      ranking_df.sort_values("Production (MT)", ascending=False)
      .head(15)
      .reset_index(drop=True)
  )

  fig_bar = px.bar(
      top_15,
      x="County",
      y="Production (MT)",
      text="Production (MT)",
      color="Production (MT)",
      color_continuous_scale="Greens",
      title=f"Top Producing Counties in {latest_year}",
  )

  fig_bar.update_layout(
      template="plotly_white",
      xaxis_title="County",
      yaxis_title="Production (Metric Tons)",
  )

  st.plotly_chart(fig_bar, use_container_width=True)
  st.dataframe(top_15, use_container_width=True)
  st.caption("The ranking records are generated using a Python generator.")

# TAB 3: HARVESTED AREA VS PRODUCTION
with tab3:
  st.subheader("Harvested Area vs Maize Production")

  if latest_year in area_pivot.columns:
    combined_df = pd.DataFrame({
        "County": prod_pivot.index,
        "Production": prod_pivot[latest_year],
        "Area": area_pivot[latest_year],
    }).dropna()
  else:
    combined_df = pd.DataFrame()

  if not combined_df.empty:
    fig_scatter = px.scatter(
        combined_df,
        x="Area",
        y="Production",
        text="County",
        size="Production",
        hover_name="County",
        title="County Land Utilization vs Harvest Output",
        labels={
            "Area": "Harvested Area (Hectares)",
            "Production": "Production (Metric Tons)",
        },
    )

    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(template="plotly_white", height=600)

    st.plotly_chart(fig_scatter, use_container_width=True)
  else:
    st.warning(
        "Area and production records are unavailable for the same year."
    )

# TAB 4: AGRONOMIC INSIGHTS
with tab4:
  st.subheader(f"🤖 Strategic Recommendations for {selected_county}")

  st.markdown(f"""
    **1. Production Trajectory**

    {selected_county} recorded approximately
    **{int(latest_prod):,} MT** of maize production
    from **{int(latest_area):,} hectares**.

    **2. Fertilizer Planning**

    Simulated fertilizer application:
    **{sim_fertilizer} kg/acre**.

    Extension officers should consider soil testing
    and appropriate fertilizer recommendations.

    **3. Rainfall Assessment**

    Simulated seasonal rainfall:
    **{sim_rainfall} mm**.

    Rainfall monitoring can support planting and
    harvesting decisions.

    **4. Seed Variety**

    Selected seed variety: **{seed_variety}**.

    Seed performance should be validated using
    local agricultural trial data.

    **5. Food Security Planning**

    Estimated 2026 production:
    **{adjusted_2026_prod:,} MT**.

    This estimate can support preliminary storage
    and food security planning.
    """)

  st.warning(
      "The seed and fertilizer multipliers are illustrative simulation"
      " assumptions, not scientifically validated agronomic response"
      " coefficients."
  )

# TAB 5: PYTHON CONCEPTS
with tab5:
  st.subheader("Python Programming Concepts Demonstrated")
  st.markdown(
      "This application demonstrates important advanced Python programming"
      " concepts."
  )

  st.markdown("### 1. Functions")
  st.code("""
def get_county_forecast(county, prod_pivot, area_pivot, yield_pivot):
    ...
    """)
  st.write(
      "Functions organize the application into reusable blocks of code."
  )

  st.markdown("### 2. Decorators")
  st.code("""
@st.cache_data
@execution_timer
def load_maize_data():
    ...
    """)
  st.write(
      "Decorators add caching and execution-time measurement without changing"
      " the original function logic."
  )

  st.markdown("### 3. Iterators")
  st.code("""
county_iter = iter(counties_list)
first_county = next(county_iter)
    """)
  st.write(
      "The iterator retrieves county records one at a time. First county:"
      f" {first_county}."
  )

  st.markdown("### 4. Generators")
  st.code("""
def county_generator(production_table, year):
    for county, production in production_table[year].items():
        yield {
            "County": county,
            "Production (MT)": production
        }
    """)
  st.write("Generators use yield to produce records one at a time.")

  st.markdown("### 5. Recursion")
  st.code("""
def recursive_total(values, index=0):
    if index >= len(values):
        return 0
    return values[index] + recursive_total(values, index + 1)
    """)
  st.write(
      "Recursive calculation of national maize production:"
      f" {int(national_total):,} MT."
  )

  st.markdown("### 6. Python Libraries")
  libraries = pd.DataFrame({
      "Library": [
          "Streamlit",
          "Pandas",
          "NumPy",
          "Plotly Express",
          "Plotly Graph Objects",
      ],
      "Purpose": [
          "Interactive dashboard",
          "Data processing and analysis",
          "Numerical calculations and regression",
          "Interactive data visualization",
          "Advanced customized charts",
      ],
  })
  st.dataframe(libraries, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    <p style='text-align:center; color:#6c757d;'>
    Kenya 47-County Maize Intelligence Engine | 
    Powered by Python, Streamlit, Pandas, NumPy and Plotly
    </p>
    """,
    unsafe_allow_html=True,
)
