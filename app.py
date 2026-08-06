import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh
import psycopg2
import pycountry

# Refresh every 60 seconds
st_autorefresh(interval=60000, key="datarefresh")

st.set_page_config(layout="wide", page_title="Global Aviation Radar & Anomaly Tracker")
st.title("🌍 Global Aviation Radar & Anomaly Tracker")

tab_live, tab_history = st.tabs(["📡 Live Radar & Anomaly Detector", "📊 Historical Trends & SQL Logs"])

# --- 1. INSTANT OFFLINE COUNTRY LIST ---
@st.cache_data
def get_country_list():
    return sorted([country.name for country in pycountry.countries])

st.sidebar.header("Dashboard Settings")
region_choice = st.sidebar.selectbox("Select Airspace Region:", get_country_list())

# --- 2. DYNAMIC GEOGRAPHIC BOUNDING BOX ---
@st.cache_data
def get_bounding_box(country_name):
    headers = {"User-Agent": "StreamlitFlightTracker/1.0"}
    url = f"https://nominatim.openstreetmap.org/search?country={country_name}&format=json"
    try:
        res = requests.get(url, headers=headers, timeout=5).json()
        if res:
            bbox = res[0]['boundingbox'] # [south, north, west, east]
            return {
                "south": float(bbox[0]),
                "north": float(bbox[1]),
                "west": float(bbox[2]),
                "east": float(bbox[3]),
                "airlabs_format": f"{bbox[0]},{bbox[2]},{bbox[1]},{bbox[3]}"
            }
    except:
        pass
    return None

# --- 3. FETCH DATA (AIRLABS WITH OPENSKY FALLBACK) ---
@st.cache_data(ttl=45) 
def fetch_flight_data(bbox):
    if not bbox:
        return [], "none"
        
    api_key = st.secrets.get("AIRLABS_API_KEY", None)
    
    # Try AirLabs First
    if api_key:
        url = f"https://airlabs.co/api/v9/flights?api_key={api_key}&bbox={bbox['airlabs_format']}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json().get('response', [])
                if data:
                    return data, "airlabs"
        except:
            pass

    # OpenSky Fallback
    opensky_url = f"https://opensky-network.org/api/states/all?lamin={bbox['south']}&lomin={bbox['west']}&lamax={bbox['north']}&lomax={bbox['east']}"
    try:
        res = requests.get(opensky_url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data and 'states' in data and data['states'] is not None:
                return data['states'], "opensky"
    except:
        pass

    return [], "none"

# --- 4. EXPLAINABILITY ENGINE ---
def generate_anomaly_reason(alt, speed, v_speed):
    reasons = []
    
    if v_speed < -15:
        reasons.append(f"Rapid Descent ({v_speed:.1f} m/s)")
    elif v_speed > 15:
        reasons.append(f"Steep Climb ({v_speed:.1f} m/s)")
        
    if alt > 3000 and speed < 100:
        reasons.append(f"Unusually Low Speed ({speed:.0f} km/h at {alt:.0f}m)")
    elif alt < 1000 and speed > 550:
        reasons.append(f"Excessive Low-Altitude Speed ({speed:.0f} km/h)")
        
    if not reasons:
        reasons.append("Multi-Variable Outlier (Combined Trajectory Anomaly)")
        
    return " | ".join(reasons)

# --- 5. WEATHER ENRICHMENT ---
def get_flight_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,precipitation"
    try:
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            current = res.json().get('current', {})
            return (
                current.get("temperature_2m", 0.0),
                current.get("wind_speed_10m", 0.0),
                current.get("precipitation", 0.0)
            )
    except:
        pass
    return 0.0, 0.0, 0.0

# --- 6. TRANSFORM & ML PIPELINE ---
def process_data(raw_data, source_type):
    if not raw_data:
        return pd.DataFrame()
        
    clean_list = []
    
    # Standardize data format regardless of source API
    if source_type == "airlabs":
        for item in raw_data:
            clean_list.append({
                'callsign': str(item.get('flight_icao', item.get('reg_number', 'UNKNOWN'))).strip(),
                'origin': item.get('flag', 'N/A'),
                'lat': item.get('lat'),
                'lon': item.get('lng'),
                'alt': item.get('alt', 0),
                'speed': item.get('speed', 0),
                'v_speed': item.get('v_speed', 0),
                'route': f"🛫 {item.get('dep_iata', 'N/A')} ➡️ 🛬 {item.get('arr_iata', 'N/A')}"
            })
    elif source_type == "opensky":
        for state in raw_data:
            clean_list.append({
                'callsign': str(state[1]).strip() if state[1] else "UNKNOWN",
                'origin': state[2],
                'lat': state[6],
                'lon': state[5],
                'alt': state[7] if state[7] is not None else 0,
                'speed': state[9] if state[9] is not None else 0,
                'v_speed': state[11] if state[11] is not None else 0,
                'route': "Route Unavailable (ADS-B Beacon)"
            })
            
    df = pd.DataFrame(clean_list)
    df = df.dropna(subset=['lat', 'lon'])
    
    if len(df) > 5:
        features = df[['alt', 'speed', 'v_speed']]
        
        # Machine Learning Anomaly Detection
        model = IsolationForest(contamination=0.03, random_state=42)
        df['anomaly_score'] = model.fit_predict(features)
        
        df['status'] = df['anomaly_score'].apply(lambda x: '⚠️ Anomaly' if x == -1 else '✅ Normal')
        df['color'] = df['status'].apply(lambda x: '#ff0000' if x == '⚠️ Anomaly' else '#0000ff')
        
        reasons, temps, winds, precips = [], [], [], []
        
        for _, row in df.iterrows():
            if row['status'] == '⚠️ Anomaly':
                # Generate exact reason
                reason = generate_anomaly_reason(row['alt'], row['speed'], row['v_speed'])
                # Fetch live weather
                t, w, p = get_flight_weather(row['lat'], row['lon'])
                
                reasons.append(reason)
                temps.append(t)
                winds.append(w)
                precips.append(p)
            else:
                reasons.append("Normal Flight Trajectory")
                temps.append(None)
                winds.append(None)
                precips.append(None)
                
        df['anomaly_reason'] = reasons
        df['temp_c'] = temps
        df['wind_speed_kmh'] = winds
        df['precipitation_mm'] = precips
    else:
        df['status'] = '✅ Normal'
        df['color'] = '#0000ff'
        df['anomaly_reason'] = 'Normal'
        df['temp_c'], df['wind_speed_kmh'], df['precipitation_mm'] = None, None, None
        
    return df

# --- 7. DATABASE LOGGING ---
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
            callsign_clean = row['callsign'] if row['callsign'] else "UNKNOWN"
            reason_full = f"[{region}] {row['anomaly_reason']} | Temp: {row.get('temp_c')}°C Wind: {row.get('wind_speed_kmh')}km/h"
            
            cursor.execute(insert_query, (
                callsign_clean,
                float(row.get('alt', 0)),
                float(row.get('speed', 0)),
                float(row.get('v_speed', 0)),
                reason_full
            ))
            
        conn.commit()
        cursor.close()
        conn.close()
        st.sidebar.success(f"SQL Log: Saved {len(anomalies_df)} anomalies.")
    except Exception as e:
        st.sidebar.error(f"SQL Error: {e}")

# --- 8. FETCH HISTORICAL SQL LOGS ---
def fetch_historical_data():
    db_uri = st.secrets.get("DB_URI", None)
    if not db_uri:
        return pd.DataFrame()
    
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
# DASHBOARD INTERFACE
# ==========================================

with tab_live:
    bbox = get_bounding_box(region_choice)
    
    with st.spinner(f'Fetching live radar telemetry over {region_choice}...'):
        raw_flights, source_type = fetch_flight_data(bbox)
        
    df = process_data(raw_flights, source_type)
    
    if df.empty:
        st.warning(f"No active flight data currently retrieved for {region_choice}. Try selecting another country or wait a minute for API refresh.")
    else:
        anomalies = df[df['status'] == '⚠️ Anomaly']
        log_anomalies_to_sql(anomalies, region_choice)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Tracking {len(df)} Active Flights over {region_choice}")
            st.map(df, latitude='lat', longitude='lon', color='color', zoom=4)
            
        with col2:
            st.subheader(f"Anomalies Detected: {len(anomalies)}")
            if not anomalies.empty:
                for _, row in anomalies.iterrows():
                    with st.expander(f"✈️ Flight {row['callsign']} ({row['origin']})"):
                        st.error(f"**Reason:** {row['anomaly_reason']}")
                        st.markdown(f"**Route Details:** {row['route']}")
                        st.markdown(f"**GPS Coordinates:** `{row['lat']:.4f}, {row['lon']:.4f}`")
                        
                        st.write("---")
                        st.write("**📡 Telemetry Values**")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Altitude", f"{row['alt']} m")
                        m2.metric("Velocity", f"{row['speed']} km/h")
                        m3.metric("Vert Rate", f"{row['v_speed']} m/s")
                        
                        st.write("---")
                        st.write("**⛅ Encrypted Live Weather**")
                        w1, w2, w3 = st.columns(3)
                        w1.metric("Temp", f"{row['temp_c']} °C")
                        w2.metric("Wind", f"{row['wind_speed_kmh']} km/h")
                        w3.metric("Precip", f"{row['precipitation_mm']} mm")
            else:
                st.success("Airspace behavior is mathematically normal.")

with tab_history:
    st.subheader("Database Analytics & Historical Anomaly Trends")
    hist_df = fetch_historical_data()
    
    if not hist_df.empty:
        st.markdown("### 1. Velocity vs. Altitude of Logged Anomalies")
        st.caption("Visualizing statistical outliers logged to Supabase PostgreSQL.")
        st.scatter_chart(hist_df, x='velocity', y='baro_altitude', color='#ff0000')
        
        st.markdown("### 2. Live Supabase Table Records")
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No historical data found in Supabase yet. As anomalies are detected on Tab 1, they will automatically populate here.")
