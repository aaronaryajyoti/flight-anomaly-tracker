import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh
import psycopg2

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, key="datarefresh")

st.set_page_config(layout="wide", page_title="Global Aviation & Weather Tracker")
st.title("🌍 Global Aviation Radar & Historical Trends")

# --- UI TABS ---
tab_live, tab_history = st.tabs(["📡 Live Global Radar", "📊 Historical Trends"])

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Dashboard Settings")
region_choice = st.sidebar.selectbox(
    "Select Airspace Region:",
    ["Global (Entire World)", "India", "United States", "Europe", "Australia"]
)

# OpenSky API boundaries
bboxes = {
    "Global (Entire World)": "",
    "India": "lamin=8&lomin=68&lamax=37&lomax=97",
    "United States": "lamin=25&lomin=-125&lamax=50&lomax=-65",
    "Europe": "lamin=35&lomin=-15&lamax=70&lomax=40",
    "Australia": "lamin=-44&lomin=113&lamax=-10&lomax=154"
}

# --- EXTRACT ---
@st.cache_data(ttl=45) 
def fetch_flight_data(region_params):
    base_url = "https://opensky-network.org/api/states/all"
    url = f"{base_url}?{region_params}" if region_params else base_url
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if data and 'states' in data and data['states'] is not None:
                return data['states']
            return []
        else:
            st.error(f"OpenSky API Rate Limit Hit (Status Code: {res.status_code}).")
            return []
    except Exception as e:
        st.error(f"Connection timeout: {e}")
        return []

# --- ENRICH (WEATHER) ---
def get_flight_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,precipitation"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            current = res.json().get('current', {})
            return current.get("temperature_2m", 0.0), current.get("wind_speed_10m", 0.0), current.get("precipitation", 0.0)
    except:
        pass
    return 0.0, 0.0, 0.0

# --- TRANSFORM (ML & CLEANING) ---
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
    
    # Cap processing for global scope to prevent memory crashes on Streamlit Cloud
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)
    
    if len(df) > 10:
        features = df[['baro_altitude', 'velocity', 'vertical_rate']]
        
        # 1% contamination ensures we only flag the most extreme outliers globally
        model = IsolationForest(contamination=0.01, random_state=42)
        df['anomaly_score'] = model.fit_predict(features)
        
        df['status'] = df['anomaly_score'].apply(lambda x: '⚠️ Anomaly' if x == -1 else '✅ Normal')
        df['color'] = df['status'].apply(lambda x: '#ff0000' if x == '⚠️ Anomaly' else '#0000ff')
        
        temps, winds, precips = [], [], []
        for _, row in df.iterrows():
            if row['status'] == '⚠️ Anomaly':
                t, w, p = get_flight_weather(row['lat'], row['lon'])
                temps.append(t)
                winds.append(w)
                precips.append(p)
            else:
                temps.append(None)
                winds.append(None)
                precips.append(None)
                
        df['temp_c'] = temps
        df['wind_speed_kmh'] = winds
        df['precipitation_mm'] = precips
    else:
        df['status'] = '✅ Normal'
        df['color'] = '#0000ff'
        df['temp_c'], df['wind_speed_kmh'], df['precipitation_mm'] = None, None, None
        
    return df

# --- LOAD (DATABASE LOGGING) ---
def log_anomalies_to_sql(anomalies_df, region):
    if anomalies_df.empty:
        return
        
    db_uri = st.secrets.get("DB_URI", None)
    if not db_uri:
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
            if not callsign_clean: callsign_clean = "UNKNOWN"
            reason = f"Zone: {region} | Temp: {row.get('temp_c')}°C | Wind: {row.get('wind_speed_kmh')}km/h"
            
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
    except:
        pass

# --- FETCH HISTORICAL DATA ---
def fetch_historical_data():
    db_uri = st.secrets.get("DB_URI", None)
    if not db_uri: return pd.DataFrame()
    
    try:
        conn = psycopg2.connect(db_uri)
        cursor = conn.cursor()
        # Pull the 500 most recent anomalies from the database
        cursor.execute("SELECT timestamp, callsign, baro_altitude, velocity, vertical_rate, anomaly_reason FROM historical_anomalies ORDER BY timestamp DESC LIMIT 500")
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)
        cursor.close()
        conn.close()
        return df
    except:
        return pd.DataFrame()

# ==========================================
# UI RENDERING
# ==========================================

# TAB 1: LIVE RADAR
with tab_live:
    with st.spinner(f'Fetching live radar data for {region_choice}...'):
        raw_states = fetch_flight_data(bboxes[region_choice])
        
    df = process_data(raw_states)
    
    if df.empty:
        st.warning(f"No active flight data retrieved. The OpenSky API might be rate-limiting global pulls.")
    else:
        anomalies = df[df['status'] == '⚠️ Anomaly']
        log_anomalies_to_sql(anomalies, region_choice)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Tracking Active Flights: {region_choice}")
            # Zoom out further for global view
            st.map(df, latitude='lat', longitude='lon', color='color', zoom=1 if region_choice == "Global (Entire World)" else 4)
            
        with col2:
            st.subheader(f"Anomalies Detected: {len(anomalies)}")
            if not anomalies.empty:
                # Individual expanding cards for each flight
                for _, row in anomalies.iterrows():
                    callsign = str(row['callsign']).strip() or "UNKNOWN"
                    with st.expander(f"✈️ Flight {callsign} ({row['origin_country']})"):
                        st.markdown(f"**GPS Coordinates:** {row['lat']:.4f}, {row['lon']:.4f}")
                        
                        st.write("---")
                        st.write("**📡 Telemetry (Anomaly Triggers)**")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Altitude", f"{row['baro_altitude']} m")
                        m2.metric("Velocity", f"{row['velocity']} m/s")
                        m3.metric("Vert Rate", f"{row['vertical_rate']} m/s")
                        
                        st.write("---")
                        st.write("**⛅ Live Weather Encounters**")
                        w1, w2, w3 = st.columns(3)
                        w1.metric("Temp", f"{row['temp_c']} °C")
                        w2.metric("Wind", f"{row['wind_speed_kmh']} km/h")
                        w3.metric("Precip", f"{row['precipitation_mm']} mm")
            else:
                st.success("Airspace behavior is mathematically normal.")

# TAB 2: HISTORICAL TRENDS
with tab_history:
    st.subheader("Database Analytics & Trends")
    hist_df = fetch_historical_data()
    
    if not hist_df.empty:
        # Chart 1: Altitude vs Velocity Scatter
        st.markdown("### 1. Velocity vs. Altitude of Past Anomalies")
        st.caption("Spotting erratic patterns: Look for data points showing exceptionally low altitude combined with high speed, or high altitude with low speed.")
        st.scatter_chart(hist_df, x='velocity', y='baro_altitude', color='#ff0000')
        
        # Table: Raw Database Logs
        st.markdown("### 2. Raw Database Logs")
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No historical data found. Waiting for anomalies to be logged...")
