# src/viz/dashboard.py
import streamlit as st
import pandas as pd
import json
import pydeck as pdk
import time
import numpy as np
from pathlib import Path
from ast import literal_eval
import h3

# --- 1. FILE PATHS (These are correct) ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets"
FORECAST_CSV = DATA_DIR / "forecast_15min_predictions.csv"
REPOSITION_CSV = DATA_DIR / "repositioning_plan.csv"
PROCESSED_DATA_CSV = DATA_DIR / "processed_data.csv" 
CSS_PATH = Path(__file__).resolve().parent / "styles.css"

def local_css(file_path):
    if file_path.exists():
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- 2. FINAL FIX FOR COORDINATE LOADING ---
@st.cache_data
def load_coordinate_mapping(path):
    """
    Loads processed_data.csv ONCE to create a mapping
    from h3_index (the H3 string) to its coordinates AND its name.
    """
    if not path.exists():
        st.error(f"Missing processed data CSV at {path}. Run pipeline.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    
    # --- THIS IS THE FIX ---
    # Use the correct column names from processed_data.csv
    coords_df = df[['h3_index', 'latitude', 'longitude', 'Area Name']].drop_duplicates()
    
    # Rename them to 'lat', 'lon', and 'zone_name' for the rest of the dashboard
    coords_df = coords_df.rename(columns={
        'latitude': 'lat', 
        'longitude': 'lon', 
        'Area Name': 'zone_name'
    })
    return coords_df

# --- 3. UPDATED FORECAST LOADING ---
def load_forecasts(forecast_path, coords_df):
    """
    Loads REAL forecast data and MERGES it with the coordinate mapping.
    """
    if not forecast_path.exists():
        st.error(f"Missing forecasts CSV at {forecast_path}. Run pipeline.")
        return pd.DataFrame()
        
    df = pd.read_csv(forecast_path, parse_dates=["next_time"])
    
    # Merge forecasts with coordinates on the H3 index
    # This will now also bring in the 'zone_name' column
    df_merged = pd.merge(df, coords_df, on="h3_index", how="left")
    
    # Rename columns to match the dashboard's expected names
    df_merged = df_merged.rename(columns={
        "h3_index": "h3",
        "pred_bookings_15min": "surge_score",
        "next_time": "timestamp"
    })

    df_merged = df_merged.dropna(subset=['lat', 'lon'])
    
    if not df_merged.empty:
        max_score = df_merged['surge_score'].max()
        if max_score > 0:
            df_merged['surge_score'] = df_merged['surge_score'] / max_score
        else:
            df_merged['surge_score'] = 0.0
    
    df_merged['contrib'] = pd.Series(dtype='object')
    return df_merged

# --- 4. UPDATED REPOSITION LOADING ---
def load_reposition(reposition_path, coords_df):
    """
    Loads REAL repositioning data and MERGES it with the coordinate mapping.
    """
    if not reposition_path.exists():
        st.warning(f"Missing repositioning CSV at {reposition_path}. Run pipeline.")
        return []

    df = pd.read_csv(reposition_path)
    
    # Merge with coordinates to get the 'to' lat/lon
    # Note: assigned_zone column contains the H3 string
    df = pd.merge(df, coords_df, left_on='assigned_zone', right_on='h3_index', how='left')
    df = df.rename(columns={'lat': 'to_lat', 'lon': 'to_lon'})
    df = df.dropna(subset=['to_lat', 'to_lon'])
    
    # Simulate 'from' coordinates by picking random zones
    if not coords_df.empty and len(df) > 0:
        # Get a random sample of coordinates from all available zones
        random_coords = coords_df.sample(n=len(df), replace=True).reset_index(drop=True)
        df['from_lat'] = random_coords['lat']
        df['from_lon'] = random_coords['lon']
    else:
        # Fallback if coords_df is empty for some reason
        df['from_lat'] = df['to_lat'] + np.random.uniform(-0.05, 0.05, size=len(df))
        df['from_lon'] = df['to_lon'] + np.random.uniform(-0.05, 0.05, size=len(df))

    return df.to_dict('records')

# --- 5. H3 POLYGON FUNCTION (This is correct) ---
def h3_polygon_geojson(h3idx):
    """
    Converts a real H3 index string into a polygon.
    """
    try:
        coords = h3.h3_to_geo_boundary(h3idx, geo_json=True)
        return [[p[1], p[0]] for p in coords]
    except:
        return []

st.set_page_config(layout="wide", page_title="EquiRide — Surge Dashboard", initial_sidebar_state="expanded")
local_css(CSS_PATH) 

st.markdown("""<div class="banner"><h1 style="margin:0; font-size:36px;">EquiRide — Surge Dashboard</h1>
<div style="opacity:0.9">Interactive heatmap • Explainability • Reposition simulation</div></div>""", unsafe_allow_html=True)

# --- 6. DATA LOADING ORDER (This is correct) ---
coords_df = load_coordinate_mapping(PROCESSED_DATA_CSV)
df = load_forecasts(FORECAST_CSV, coords_df)
reposition = load_reposition(REPOSITION_CSV, coords_df)

if df.empty:
    st.error("No data to display. Check pipeline output files.")
    st.stop()

# --- 7. TIME SELECTION (This is correct) ---
unique_times = sorted(df['timestamp'].unique())
if not unique_times:
    st.error("No forecast data to display.")
    st.stop()

if len(unique_times) > 1:
    selected_time = st.sidebar.select_slider("Forecast time", options=unique_times, value=unique_times[0])
    st.sidebar.write("Selected:", selected_time)
else:
    selected_time = unique_times[0]
    st.sidebar.info(f"Showing forecast for: {selected_time}")


# --- 8. MAP PREPARATION (This is correct) ---
selected_df = df[df['timestamp'] == selected_time]
if selected_df.empty:
    st.warning("No forecast rows for this timestamp.")
else:
    layers = []
    
    all_scores_zero = selected_df['surge_score'].max() == 0

    if not all_scores_zero:
        heat_data = [{"position":[r['lon'], r['lat']], "weight": float(r['surge_score'])} for _, r in selected_df.iterrows()]
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=heat_data,
            get_position="position",
            get_weight="weight",
            radiusPixels=80
        ))

    scatter_df = pd.DataFrame([{"lon":r['lon'], "lat":r['lat'], "surge":float(r['surge_score']), "h3":r['h3']} for _, r in selected_df.iterrows()])
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=scatter_df,
        get_position=["lon","lat"],
        get_radius=200, 
        get_fill_color="[255*surge, 80*(1-surge), 200*(1-surge), 150]", 
        pickable=True
    ))

    hex_polys = []
    for _, r in selected_df.iterrows():
        poly = h3_polygon_geojson(r['h3'])
        if poly:
            hex_polys.append({"polygon": poly, "surge": float(r['surge_score'])})
            
    if hex_polys:
        hex_df = pd.DataFrame(hex_polys)
        layers.append(pdk.Layer(
            "PolygonLayer",
            data=hex_df,
            get_polygon="polygon",
            get_fill_color="[255*surge, 60*(1-surge), 200*(1-surge)]",
            pickable=True,
            stroked=True,
            get_line_color=[50,50,50]
        ))

    # reposition layers
    if len(reposition) > 0:
        r_df = pd.DataFrame(reposition)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=r_df,
            get_position=["from_lon","from_lat"],
            get_radius=50,
            get_fill_color=[30,144,255],
            pickable=True
        ))
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=r_df,
            get_position=["to_lon","to_lat"],
            get_radius=80,
            get_fill_color=[50,205,50],
            pickable=True
        ))
        path_data = []
        for row in reposition:
            path_data.append({"path":[[row['from_lon'], row['from_lat']], [row['to_lon'], row['to_lat']]], "driver": row['driver_id']})
        layers.append(pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            width_scale=10,
            width_min_pixels=2,
            get_color=[160, 100, 255]
        ))

    mid = (selected_df.iloc[0]['lat'], selected_df.iloc[0]['lon'])
    view_state = pdk.ViewState(latitude=mid[0], longitude=mid[1], zoom=13, pitch=30)
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10")
    st.pydeck_chart(deck)

# --- 9. FINAL FIX FOR SIDEBAR ---
st.sidebar.header("Zone Explainability")
# Use the 'zone_name' column we just merged in
zone_name_options = sorted(list(selected_df['zone_name'].unique()))

if zone_name_options:
    # Show the human-readable names in the dropdown
    chosen_name = st.sidebar.selectbox("Zone", zone_name_options)
    
    # Filter the data by the chosen name
    row = selected_df[selected_df['zone_name'] == chosen_name].iloc[0]
    
    st.sidebar.metric("Surge score (Normalized)", f"{row['surge_score']:.2f}")
    # Also display the H3 index for reference
    st.sidebar.caption(f"H3 Index: {row['h3']}")


# Simulation
st.header("Repositioning simulation (demo)")
if st.button("Run simulation"):
    placeholder = st.empty()
    steps = 8
    for s in range(1, steps+1):
        inter = []
        for d in reposition:
            lat = d['from_lat'] + (d['to_lat'] - d['from_lat'])*(s/steps)
            lon = d['from_lon'] + (d['to_lon'] - d['from_lon'])*(s/steps)
            inter.append({"driver": d['driver_id'], "lat": lat, "lon": lon})
        df_step = pd.DataFrame(inter)
        layer = pdk.Layer("ScatterplotLayer", data=df_step, get_position=["lon","lat"], get_radius=100, get_fill_color=[255,200,100])
        view = pdk.ViewState(latitude=mid[0], longitude=mid[1], zoom=13, pitch=30)
        deck = pdk.Deck(layers=[layer], initial_view_state=view, map_style="mapbox://styles/mapbox/dark-v10")
        placeholder.pydeck_chart(deck)
        time.sleep(0.5)

st.header("Alerts (demo)")
to_phone = st.text_input("Send test alert to", value="+911234567890")
msg = st.text_area("Message", value="Demo: Surge predicted. Please reposition.")
if st.button("Send test alert"):
    try:
        from alert_stub import send_alert_stub 
        success, resp = send_alert_stub("", "", "", to_phone, msg)
        if success:
            st.success("Alert sent (stub). Check app logs/console.")
        else:
            st.error("Alert failed.")
    except ImportError:
        st.error("Could not find 'alert_stub.py'. Make sure it's in the 'src/viz/' folder.")