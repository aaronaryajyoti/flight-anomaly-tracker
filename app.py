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

# 1. EXTRACT: Pull live states over the USA
@st.cache_data(ttl=45) # Increased cache time to reduce API ban risk
def fetch_flight_data():
    # Bounding box roughly covering the Continental US
    url = "https://opensky-network.org/api/states/all?lamin=25&lomin=-125&lamax=50&lomax=-65"
    
    try:
        # A 10-second timeout prevents the app from hanging infinitely
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

# 2. TRANSFORM & ML: Format data and run anomaly detection
def process_data(states):
    if not states:
        return pd.DataFrame()
        
    # OpenSky API returns arrays, these are the corresponding columns
    cols = ['icao24', 'callsign', 'origin_country', 'time_position', 'last_contact', 
            'lon', 'lat', 'baro_altitude', 'on_ground', 'velocity', 'true_track', 
            'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source', 'category']
    
    df = pd.DataFrame(states, columns=cols)
    
    # Clean: Filter for airborne planes with complete state vectors
    df = df[(df['on_ground'] == False) & 
            df['baro_altitude'].notnull() & 
            df['velocity'].notnull() & 
            df['vertical_rate'].notnull()]
    
    # Drop planes with impossible coordinates
    df = df.dropna(subset=['lat', 'lon'])
    
    if len(df) > 10:
        # ML Feature Selection: We want planes doing unusual physical maneuvers
        features = df[['baro_altitude', 'velocity', 'vertical_rate']]
        
        # Train model to find the 2% most unusual flight patterns
        model = IsolationForest(contamination=0.02, random_state=42)
        df['anomaly_score'] = model.fit_predict(features)
        
        # Format for dashboard
        df['status'] = df['anomaly_score'].apply(lambda x: '⚠️ Anomaly' if x == -1 else '✅ Normal')
        # Red dots for anomalies, blue for normal
        df['color'] = df['status'].apply(lambda x: '#ff0000' if x == '⚠️ Anomaly' else '#0000ff')
    else:
        df['status'] = '✅ Normal'
        df['color'] = '#0000ff'
        
    return df

# Run Pipeline
with st.spinner('Fetching live radar data from OpenSky...'):
    raw_states = fetch_flight_data()
    
df = process_data(raw_states)

if df.empty:
    st.warning("No data retrieved. OpenSky API might be rate-limiting you. The dashboard will automatically retry in 60 seconds.")
    st.stop()

# 3. LOAD & VISUALIZE
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Tracking {len(df)} Active Flights")
    # Plot all flights on a map
    st.map(df, latitude='lat', longitude='lon', color='color', zoom=3)

with col2:
    anomalies = df[df['status'] == '⚠️ Anomaly']
    st.subheader(f"Anomalies Detected: {len(anomalies)}")
    
    if not anomalies.empty:
        # Format the anomalies table for readability
        display_cols = ['callsign', 'baro_altitude', 'velocity', 'vertical_rate']
        st.dataframe(anomalies[display_cols].reset_index(drop=True), use_container_width=True)
        st.caption("Look for extreme vertical rates (rapid climbs/dives) or unusually low speeds at high altitudes.")
    else:
        st.success("Airspace behavior is normal.")
