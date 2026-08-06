import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh
import psycopg2

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, key="datarefresh")

st.set_page_config(layout="wide", page_title="Live Aviation Radar & Anomaly Tracker")
st.title("✈️ Live Aviation Radar & Anomaly Tracker")

# 1. User Input: Region Selection
st.sidebar.header("Dashboard Settings")
region_choice = st.sidebar.selectbox(
    "Select Airspace Region:",
    ["India", "United States", "Europe", "Australia"]
)

# Bounding boxes mapping for the OpenSky API
bboxes = {
    "India": "lamin=8&lomin=68&lamax=37&lomax=97",
    "United States": "lamin=25&lomin=-125&lamax=50&lomax=-65",
    "Europe": "lamin=35&lomin=-15&lamax=70&lomax=40",
    "Australia": "lamin=-44&lomin=113&lamax=-10&lomax=154"
}

# 2. EXTRACT: Pull live states from OpenSky API
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
            st.error(f"OpenSky API Rate Limit Hit (Status Code: {res.status_code}). Retrying in 60s...")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Connection timeout or error: {e}")
        return []

# 3. ENRICH: Fetch live weather from Open-Meteo for flagged anomalies
def get_flight_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,precipitation"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            current = res.json().get('current', {})
            return current.get("wind_speed_10m", 0.0), current.get("precipitation", 0.0)
    except:
        pass
    return 0.0, 0.0

# 4. TRANSFORM & ML: Format data, predict anomalies, and enrich weather
def process_data(states):
    if not states:
        return pd.DataFrame()
        
    cols = ['icao24', 'callsign', 'origin_country', 'time_position', 'last_contact', 
            'lon', 'lat', 'baro_altitude', 'on_ground', 'velocity', 'true_track', 
            'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source', 'category']
    
    df = pd.DataFrame(states, columns=cols[:len(states[0])])
    
    # Filter airborne planes with valid metrics
    df = df[(df['on_ground'] == False) & 
            df['baro_altitude'].notnull() & 
            df['velocity'].notnull() & 
            df['vertical_rate'].notnull()]
    
    df = df.dropna(subset=['lat', 'lon'])
    
    if len(df) > 10:
        features = df[['baro_altitude', 'velocity', 'vertical_rate']]
        
        # Train Isolation Forest model
        model = IsolationForest(contamination=0.02, random_state=42)
        df['anomaly_score'] = model.fit_predict(features)
        
        df['status'] = df['anomaly_score'].apply(lambda x: '⚠️ Anomaly' if x == -1 else '✅ Normal')
        df['color'] = df['status'].apply(lambda x: '#ff0000' if x == '⚠️ Anomaly' else '#0000ff')
        
        # Weather Enrichment for anomalies
        wind_speeds, precipitations = [], []
        for _, row in df.iterrows():
            if row['status'] == '⚠️ Anomaly':
                wind, precip = get_flight_weather(row['lat'], row['lon'])
                wind_speeds.append(wind)
                precipitations.append(precip)
            else:
                wind_speeds.append(None)
                precipitations.append(None)
                
        df['wind_speed_kmh'] = wind_speeds
        df['precipitation_mm'] = precipitations
    else:
        df['status'] = '✅ Normal'
        df['color'] = '#0000ff'
        df['wind_speed_kmh'] = None
        df['precipitation_mm'] = None
        
    return df

# 5. LOAD TO SQL: Persist detected anomalies to Supabase
def log_anomalies_to_sql(anomalies_df, region):
    if anomalies_df.empty:
        return
        
    db_uri = st.secrets.get("DB_URI", None)
    if not db_uri:
        st.sidebar.info("Database URI not found in Secrets. Running without SQL logging.")
        return

    try:
        conn = psycopg2.connect(db_uri)
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO historical_anomalies (callsign, baro_altitude, velocity, vertical_rate, anomaly_reason)
            VALUES (%s, %s, %s, %s, %s);
        """
        
        for _, row in anomalies_df.iterrows():
            callsign_clean = str(row.get('callsign', 'N/A')).strip()
            if not callsign_clean:
                callsign_clean = "UNKNOWN"
                
            wind = row.get('wind_speed_kmh')
            precip = row.get('precipitation_mm')
            reason = f"Zone: {region} | Wind: {wind} km/h | Precip: {precip} mm"
            
            cursor.execute(insert_query, (
                callsign_clean,
                float(row.get('baro_altitude', 0)),
                float(row.get('velocity', 0)),
                float(row.get('vertical_rate', 0)),
                reason
            ))
            
        conn.commit()
        cursor.close()
        conn.close()
        st.sidebar.success(f"Log: Saved {len(anomalies_df)} anomalies to SQL database.")
    except Exception as e:
        st.sidebar.error(f"SQL Log Status: {e}")

# Run Pipeline
with st.spinner(f'Fetching live radar data for {region_choice}...'):
    raw_states = fetch_flight_data(bboxes[region_choice])
    
df = process_data(raw_states)

if df.empty:
    st.warning(f"No active flight data retrieved for {region_choice}. The API might be rate-limiting or updating.")
    st.stop()

# LOAD TO SQL
anomalies = df[df['status'] == '⚠️ Anomaly']
log_anomalies_to_sql(anomalies, region_choice)

# 6. VISUALIZE
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Tracking {len(df)} Active Flights over {region_choice}")
    st.map(df, latitude='lat', longitude='lon', color='color', zoom=3)

with col2:
    st.subheader(f"Anomalies Detected: {len(anomalies)}")
    
    if not anomalies.empty:
        display_cols = ['callsign', 'baro_altitude', 'velocity', 'vertical_rate', 'wind_speed_kmh', 'precipitation_mm']
        st.dataframe(anomalies[display_cols].reset_index(drop=True), use_container_width=True)
        st.caption("Includes live Open-Meteo weather parameters for flagged flight coordinates.")
    else:
        st.success("Airspace behavior is normal.")
