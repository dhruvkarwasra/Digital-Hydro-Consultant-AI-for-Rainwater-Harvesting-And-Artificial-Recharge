import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- PAGE CONFIGURATION ---
# This must be the very first Streamlit command
st.set_page_config(page_title="Digital Hydro-Consultant", page_icon="💧", layout="wide")

# --- CONSTANTS ---
DATA_FILE = "INDIA_HYDROLOGY_FINAL_LABELED.parquet"
MODEL_FILE = "hydro_ultimate_rf_model.pkl"

FEATURES = [
    "annual_avg_mm",
    "recent_10yr_avg_mm",
    "cv_reliability",
    "avg_max_dry_days",
    "p95_daily_mm",
    "peak_daily_mm",
    "trend_dry_days_per_year",
    "trend_rainy_days_per_year",
    "trend_sdii_intensity",
    "trend_p95_intensity",
    "trend_peak_intensity",
    "design_15min_filter_intensity_mm",
    "design_15min_overflow_intensity_mm",
    "avg_sand_pct",
    "avg_clay_pct",
    "ELEVATION_MEAN",
    "RELIEF_M",
    "RUGGEDNESS_TRI",
    "SLOPE_DEG",
    "CURVATURE",
]

ZONE_COLORS = {
    "Zone 1: Dual System (High AR + High RWH)": "#00FFAA",
    "Zone 2: Primary Recharge (Favorable Soil/Terrain)": "#00AAFF",
    "Zone 3: Surface Storage Priority (Impermeable Soil)": "#FFBB00",
    "Zone 4: Storage Priority (Hilly/Rugged Terrain - AR Unsafe)": "#FF3333",
    "Zone 5: Mixed/Moderate Potential": "#AAAAAA",
}


# --- BACKEND FUNCTIONS ---
@st.cache_resource
def load_system():
    """Loads the massive dataset and ML model into RAM exactly once."""
    df = pd.read_parquet(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    return df, model


def get_nearest_grid(target_lat, target_lon, df):
    """Snaps user coordinates to the nearest 25km pre-processed geographical grid."""
    distances = np.sqrt(
        (df["LATITUDE"] - target_lat) ** 2 + (df["LONGITUDE"] - target_lon) ** 2
    )
    nearest_idx = distances.idxmin()
    return df.loc[nearest_idx]


# --- MAIN APPLICATION ---
def main():
    st.title("🌊 Digital Hydro-Consultant")
    st.markdown("### AI-Powered Rainwater Harvesting & Artificial Recharge Assessment")
    st.markdown("---")

    # 1. System Boot
    try:
        df_master, ai_model = load_system()
    except FileNotFoundError:
        st.error(
            "❌ Critical Error: Dataset or Model file not found. Ensure they are in the directory."
        )
        return

    # 2. Session State Initialization
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False

    if "lat_input" not in st.session_state:
        st.session_state.lat_input = 29.6000

    if "lon_input" not in st.session_state:
        st.session_state.lon_input = 74.3000

    # 3. Intercept Map Clicks (Must happen before the sidebar is drawn)
    if "hydro_map" in st.session_state:
        map_data = st.session_state.hydro_map
        if map_data and "selection" in map_data and map_data["selection"]["points"]:
            clicked_lat = map_data["selection"]["points"][0]["lat"]
            clicked_lon = map_data["selection"]["points"][0]["lon"]

            # If the user clicked a new dot, update the sidebar and force a new report
            if (
                st.session_state.lat_input != clicked_lat
                or st.session_state.lon_input != clicked_lon
            ):
                st.session_state.lat_input = clicked_lat
                st.session_state.lon_input = clicked_lon
                st.session_state.report_generated = True

    # 4. Sidebar UI
    with st.sidebar:
        st.header("📍 Location Input")
        st.markdown("Enter coordinates or click a point on the map below.")

        input_lat = st.number_input(
            "Latitude", format="%.4f", key="lat_input", step=0.01
        )
        input_lon = st.number_input(
            "Longitude", format="%.4f", key="lon_input", step=0.01
        )

        st.markdown("---")
        st.header("🏠 Property Details")
        input_roof_area = st.number_input(
            "Rooftop Area (sq meters)",
            value=150.0,
            min_value=10.0,
            max_value=10000.0,
            step=10.0,
        )

        st.markdown("---")
        if st.button(
            "Generate Diagnostic Report", type="primary", use_container_width=True
        ):
            st.session_state.report_generated = True

    # 5. Core Execution Engine
    if st.session_state.report_generated:
        # --- ML Inference ---
        grid = get_nearest_grid(input_lat, input_lon, df_master)
        X_input = grid[FEATURES].to_frame().T
        predicted_zone = ai_model.predict(X_input)[0]

        # --- Extract Physical Variables ---
        rain = grid["recent_10yr_avg_mm"]
        peak_storm = grid["peak_daily_mm"]
        dry_days = grid["avg_max_dry_days"]
        clay = grid["avg_clay_pct"]
        sand = grid["avg_sand_pct"]
        slope = grid["SLOPE_DEG"]
        harvest_liters = rain * input_roof_area * 0.85

        # --- AI Engineering Diagnostics ---
        if "Zone 4" in predicted_zone:
            st.error(f"🏔️ **{predicted_zone}**")
            structure = "Above-Ground Cisterns / Contour Trenches"
            warning = f"**HIGH RUNOFF RISK:** Slope is {slope:.1f}°. Do not build open pits. Landslide danger."
        elif "Zone 3" in predicted_zone:
            st.warning(f"🧱 **{predicted_zone}**")
            structure = "Surface Ponds / Deep Recharge Shafts"
            warning = f"**LOW PERMEABILITY:** Clay content is extremely high ({clay:.1f}%). Requires deep injection."
        elif "Zone 1" in predicted_zone or "Zone 2" in predicted_zone:
            st.success(f"✅ **{predicted_zone}**")
            structure = "Standard Recharge Pit with Desilting Chamber"
            if peak_storm > 150:
                warning = f"**FLASH FLOOD RISK:** Extreme historical storm ({peak_storm:.0f}mm/day). Upsize pipes."
            else:
                warning = (
                    f"Excellent infiltration expected. High Sand content ({sand:.1f}%)."
                )
        else:
            st.info(f"⚖️ **{predicted_zone}**")
            structure = "Hybrid System (Filter -> Tank -> Soak Pit)"
            warning = f"Moderate terrain. Longest dry spell is {dry_days:.0f} days. Size primary tank accordingly."

        # --- Render Diagnostics Dashboard ---
        st.subheader("👷 Expert Engineering Recommendation")
        st.markdown(f"**Recommended Structure:** {structure}")
        st.markdown(f"**Engineering Note:** {warning}")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🌊 Hydrology")
            st.metric("Reliable Rainfall", f"{rain:.0f} mm/yr")
            st.metric("Max Dry Spell", f"{dry_days:.0f} Days")
        with col2:
            st.markdown("#### 🪨 Geotechnical")
            st.metric("Terrain Slope", f"{slope:.1f}°")
            st.metric("Soil Composition", f"{sand:.0f}% Sand | {clay:.0f}% Clay")
        with col3:
            st.markdown("#### ⚡ Design Loads")
            st.metric(
                "Est. Roof Harvest",
                f"{harvest_liters:,.0f} L/yr",
                help=f"Based on your {input_roof_area} m² roof.",
            )
            st.metric("Extreme Storm Load", f"{peak_storm:.0f} mm/day")

        st.markdown("---")

        # --- Render Interactive Simulator ---
        st.subheader("🧮 Interactive Tank Sizer & Overflow Simulator")
        daily_usage = st.slider(
            "Daily Household Water Usage (Liters)",
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
        )

        dry_spell_demand = dry_days * daily_usage
        overflow = harvest_liters - dry_spell_demand

        col_sim1, col_sim2, col_sim3 = st.columns(3)
        with col_sim1:
            st.metric("Total Harvest Potential", f"{harvest_liters:,.0f} L")
        with col_sim2:
            st.metric(
                f"Dry Spell Demand ({dry_days:.0f} days)", f"{dry_spell_demand:,.0f} L"
            )
        with col_sim3:
            if overflow > 0:
                st.metric(
                    "Excess Overflow",
                    f"{overflow:,.0f} L",
                    delta="Must be safely disposed",
                )
            else:
                st.metric(
                    "Water Deficit",
                    f"{abs(overflow):,.0f} L",
                    delta="- Requires external tanker",
                    delta_color="inverse",
                )

        # Dynamic Safety Warnings based on AI output and User Simulator Math
        if overflow > 0:
            if "Zone 3" in predicted_zone:
                st.error(
                    f"🚨 **CLAY TRAP FLOOD WARNING:** You have **{overflow:,.0f} Liters** of overflow. {clay:.1f}% clay soil will not absorb this. Route to Deep Injection Shafts."
                )
            elif "Zone 4" in predicted_zone:
                st.warning(
                    f"⚠️ **LANDSLIDE RISK:** You have **{overflow:,.0f} Liters** of overflow. Do not inject into the steep {slope:.1f}° slope. Route via Contour Trenches."
                )
            else:
                st.success(
                    f"✅ **Safe Recharge:** You have **{overflow:,.0f} Liters** of overflow. The {sand:.1f}% sandy soil can safely absorb this via soak pit."
                )
        else:
            st.warning(
                f"🏜️ **SYSTEM DRY:** Your {input_roof_area}m² roof cannot sustain {daily_usage}L/day during the {dry_days:.0f}-day drought."
            )

        st.markdown("---")

        # --- Render Plotly Interactive Map ---
        st.subheader("🗺️ Hydrological Zone Mapping")

        # Overlay Toggle
        show_overlay = st.toggle(
            "Show AI Hydrological Grid Overlay",
            value=True,
            help="Turn this off to hide the 5,000 data points and view the clean geographical base map.",
        )

        st.markdown(
            f"**Grid Match:** System aligned to Lat {grid['LATITUDE']:.2f}, Lon {grid['LONGITUDE']:.2f}"
        )

        # Data Routing based on toggle
        plot_df = df_master if show_overlay else df_master.iloc[0:0]

        # Build Map
        fig = px.scatter_mapbox(
            plot_df,
            lat="LATITUDE",
            lon="LONGITUDE",
            color="EXPERT_ZONE",
            color_discrete_map=ZONE_COLORS,
            category_orders={
                "EXPERT_ZONE": list(ZONE_COLORS.keys())
            },  # 👈 THE FIX: Forces strict 1-5 sorting
            hover_name="EXPERT_ZONE",
            hover_data={
                "LATITUDE": False,
                "LONGITUDE": False,
                "EXPERT_ZONE": False,
                "recent_10yr_avg_mm": True,
                "SLOPE_DEG": True,
            },
            zoom=4,
            center={"lat": float(grid["LATITUDE"]), "lon": float(grid["LONGITUDE"])},
            mapbox_style="carto-darkmatter",
        )

        # Add Marker for current location
        fig.add_scattermapbox(
            lat=[grid["LATITUDE"]],
            lon=[grid["LONGITUDE"]],
            mode="markers",
            marker=dict(size=15, color="white", symbol="circle"),
            name="Your Location",
        )

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        # Render Map with click-detection and zoom unlocked
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True},
            on_select="rerun",
            selection_mode="points",
            key="hydro_map",
        )


if __name__ == "__main__":
    main()
