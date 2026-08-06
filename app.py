import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh
import time

# Auto-refresh every 60 seconds.
st_autorefresh(interval=60000, key="datarefresh")

st.set_page_config(layout="wide", page_title="Live Flight Anomaly Tracker")
st.title("✈️ Live Aviation Radar & Anomaly Tracker")

# 1. User Input: Region Selection
st.sidebar.header("Dashboard Settings")
region_choice = st.sidebar.selectbox(
    "Select Airspace Region:",
    ["India", "United States", "Europe", "Australia"]
)

# Bounding boxes mapping for the API
bboxes = {
    "India": "lamin=8&lomin=68&lamax=37&lomax=97",
    "United States": "lamin=25&lomin=-125&lamax=50&lomax=-65",
    "Europe": "lamin=35&lomin=-15&lamax=70&lomax=40",
    "Australia": "lamin=-44&lomin=113&lamax=-10&lomax=154"
}

# 2. EXTRACT: Pull live states for the selected region
@st.cache_data(ttl=45) 
def fetch_flight_data(region_params):
    url = f"https://opensky-network.org/api/states/all?{region_params}"
    
    try:
        res = requests.get(url, timeout=10)
        
        if res.status_code == 200:
            data = res.json()
            if data and 'states' in data and data['states'] is not None:
                return data['states']
            else:
                return []
        else:
            st.error(f"OpenSky API Rate Limit Hit (Status Code: {res.status_code}). Please wait a minute.")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection timeout or error: {e}")
        return []

# 3. TRANSFORM & ML: Format data and run anomaly detection
def process_data(states):
    if not states:
        return pd.DataFrame()
        
    cols = ['icao24', 'callsign', 'origin_country', 'time_position', 'last_contact', 
            'lon', 'lat', 'baro_altitude', 'on_ground', 'velocity', 'true_track', 
            'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source', 'category']
    
    df = pd.DataFrame(states, columns=cols[:len(states[0])])
    
    df = df[(df['on_ground'] == False) & 
            df['baro_altitude'].notnull() & 
            df['velocity'].notnull() & 
            df['vertical_rate'].notnull()]
    
    df = df.dropna(subset=['lat', 'lon'])
    
    if len(df) > 10:
        features = df[['baro_altitude', 'velocity', 'vertical_rate']]
        
        model = IsolationForest(contamination=0.02, random_state=42)
        df['anomaly_score'] = model.fit_predict(features)
        
        df['status'] = df['anomaly_score'].apply(lambda x: '⚠️ Anomaly' if x == -1 else '✅ Normal')
        df['color'] = df['status'].apply(lambda x: '#ff0000' if x == '⚠️ Anomaly' else '#0000ff')
    else:
        df['status'] = '✅ Normal'
        df['color'] = '#0000ff'
        
    return df

# Run Pipeline based on the user's region choice
with st.spinner(f'Fetching live radar data for {region_choice}...'):
    raw_states = fetch_flight_data(bboxes[region_choice])
    
df = process_data(raw_states)

if df.empty:
    st.warning(f"No data retrieved for {region_choice}. The API might be rate-limiting or there are no active flights in this zone.")
    st.stop()

# 4. LOAD & VISUALIZE
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Tracking {len(df)} Active Flights over {region_choice}")
    st.map(df, latitude='lat', longitude='lon', color='color', zoom=3)

with col2:
    anomalies = df[df['status'] == '⚠️ Anomaly']
    st.subheader(f"Anomalies Detected: {len(anomalies)}")
    
    if not anomalies.empty:
        display_cols = ['callsign', 'baro_altitude', 'velocity', 'vertical_rate']
        st.dataframe(anomalies[display_cols].reset_index(drop=True), use_container_width=True)
        st.caption("Look for extreme vertical rates (rapid climbs/dives) or unusually low speeds at high altitudes.")
    else:
        st.success("Airspace behavior is normal.")
