import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh
import psycopg2

st_autorefresh(interval=60000, key="datarefresh")

st.set_page_config(layout="wide", page_title="Global Aviation Tracker")
st.title("🌍 Global Aviation Radar & Historical Trends")

tab_live, tab_history = st.tabs(["📡 Live Global Radar", "📊 Historical Trends"])

# --- API 1: Fetch All 216 Countries Dynamically ---
@st.cache_data(ttl=86400)
def get_country_list():
    try:
        res = requests.get("https://restcountries.com/v3.1/all", timeout=5)
        countries = [c['name']['common'] for c in res.json()]
        return sorted(countries)
    except:
        return ["India", "United States", "United Kingdom", "Australia"]

st.sidebar.header("Dashboard Settings")
region_choice = st.sidebar.selectbox("Select Airspace Region:", get_country_list())

# --- API 2: Calculate Bounding Box via Nominatim ---
@st.cache_data
def get_bounding_box(country_name):
    headers = {"User-Agent": "StreamlitFlightTracker/1.0"}
    url = f"https://nominatim.openstreetmap.org/search?country={country_name}&format=json"
    try:
        res = requests.get(url, headers=headers, timeout=5).json()
        if res:
            bbox = res[0]['boundingbox']
            # Nominatim format: [south, north, west, east]
            # AirLabs format: south,west,north,east
            return f"{bbox[0]},{bbox[2]},{bbox[1]},{bbox[3]}"
    except:
        pass
    return None

# --- API 3: Extract Flight Data via AirLabs ---
@st.cache_data(ttl=45) 
def fetch_flight_data(bbox_str):
    api_key = st.secrets.get("AIRLABS_API_KEY", None)
    if not api_key or not bbox_str:
        return []
    
    url = f"https://airlabs.co/api/v9/flights?api_key={api_key}&bbox={bbox_str}"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            return res.json().get('response', [])
        else:
            st.error(f"AirLabs API Error: {res.status_code}. Check your API Key.")
            return []
    except Exception as e:
        st.error(f"Connection error: {e}")
        return []

# --- API 4: Enrich with Live Weather via Open-Meteo ---
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
def process_data(flights):
    if not flights:
        return pd.DataFrame()
        
    df = pd.DataFrame(flights)
    
    # Filter valid flights and map AirLabs columns to our standard format
    df = df.dropna(subset=['lat', 'lng', 'alt', 'speed', 'v_speed'])
    
    if len(df) > 10:
        features = df[['alt', 'speed', 'v_speed']]
        
        model = IsolationForest(contamination=0.02, random_state=42)
        df['anomaly_score'] = model.fit_predict(features)
        
        df['status'] = df['anomaly_score'].apply(lambda x: '⚠️ Anomaly' if x == -1 else '✅ Normal')
        df['color'] = df['status'].apply(lambda x: '#ff0000' if x == '⚠️ Anomaly' else '#0000ff')
        
        temps, winds, precips = [], [], []
        for _, row in df.iterrows():
            if row['status'] == '⚠️ Anomaly':
                t, w, p = get_flight_weather(row['lat'], row['lng'])
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
    if anomalies_df.empty: return
    db_uri = st.secrets.get("DB_URI", None)
    if not db_uri: return

    try:
        conn = psycopg2.connect(db_uri)
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO historical_anomalies (callsign, baro_altitude, velocity, vertical_rate, anomaly_reason)
            VALUES (%s, %s, %s, %s, %s);
        """
        
        for _, row in anomalies_df.iterrows():
            callsign_clean = str(row.get('flight_icao', 'UNKNOWN')).strip()
            
            # Pack route and weather into the anomaly reason to avoid breaking the SQL table structure
            route = f"{row.get('dep_iata', 'N/A')}->{row.get('arr_iata', 'N/A')}"
            reason = f"Zone: {region} | Route: {route} | Temp: {row.get('temp_c')}C | Wind: {row.get('wind_speed_kmh')}km/h"
            
            cursor.execute(insert_query, (
                callsign_clean,
                float(row.get('alt', 0)),
                float(row.get('speed', 0)),
                float(row.get('v_speed', 0)),
                reason
            ))
            
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.sidebar.error(f"SQL Error: {e}")

# --- FETCH HISTORICAL DATA ---
def fetch_historical_data():
    db_uri = st.secrets.get("DB_URI", None)
    if not db_uri: return pd.DataFrame()
    
    try:
        conn = psycopg2.connect(db_uri)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, callsign, baro_altitude, velocity, vertical_rate, anomaly_reason FROM historical_anomalies ORDER BY timestamp DESC LIMIT 500")
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=columns)
        cursor.close()
        conn.close()
        return df
    except:
        return pd.DataFrame()

# ==========================================
# UI RENDERING
# ==========================================

with tab_live:
    bbox_str = get_bounding_box(region_choice)
    
    with st.spinner(f'Analyzing airspace over {region_choice}...'):
        raw_flights = fetch_flight_data(bbox_str)
        
    df = process_data(raw_flights)
    
    if df.empty:
        st.warning(f"No active flight data retrieved for {region_choice}. There may be no commercial flights currently overhead, or the API is rate-limiting.")
    else:
        anomalies = df[df['status'] == '⚠️ Anomaly']
        log_anomalies_to_sql(anomalies, region_choice)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Tracking {len(df)} Active Flights over {region_choice}")
            st.map(df, latitude='lat', longitude='lng', color='color', zoom=3)
            
        with col2:
            st.subheader(f"Anomalies Detected: {len(anomalies)}")
            if not anomalies.empty:
                for _, row in anomalies.iterrows():
                    callsign = str(row.get('flight_icao', 'UNKNOWN')).strip()
                    with st.expander(f"✈️ Flight {callsign}"):
                        
                        # New Route Features from AirLabs
                        st.markdown(f"**Route:** 🛫 {row.get('dep_iata', 'N/A')} ➡️ 🛬 {row.get('arr_iata', 'N/A')}")
                        st.markdown(f"**GPS Coordinates:** {row['lat']:.4f}, {row['lng']:.4f}")
                        
                        st.write("---")
                        st.write("**📡 Telemetry (Anomaly Triggers)**")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Altitude", f"{row['alt']} m")
                        m2.metric("Velocity", f"{row['speed']} km/h")
                        m3.metric("Vert Rate", f"{row['v_speed']} m/s")
                        
                        st.write("---")
                        st.write("**⛅ Live Weather Encounters**")
                        w1, w2, w3 = st.columns(3)
                        w1.metric("Temp", f"{row['temp_c']} °C")
                        w2.metric("Wind", f"{row['wind_speed_kmh']} km/h")
                        w3.metric("Precip", f"{row['precipitation_mm']} mm")
            else:
                st.success("Airspace behavior is mathematically normal.")

with tab_history:
    st.subheader("Database Analytics & Trends")
    hist_df = fetch_historical_data()
    
    if not hist_df.empty:
        st.markdown("### 1. Velocity vs. Altitude of Past Anomalies")
        st.caption("Spotting erratic patterns: Look for data points showing exceptionally low altitude combined with high speed, or high altitude with low speed.")
        st.scatter_chart(hist_df, x='velocity', y='baro_altitude', color='#ff0000')
        
        st.markdown("### 2. Raw Database Logs")
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No historical data found. Waiting for anomalies to be logged...")
