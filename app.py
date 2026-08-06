import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh
import psycopg2
import pycountry
import airportsdata

# Refresh every 60 seconds
st_autorefresh(interval=60000, key="datarefresh")

st.set_page_config(layout="wide", page_title="Autonomous Global Aviation Intelligence Hub")
st.title("🛰️ Autonomous Global Aviation Intelligence Hub")

tab_live, tab_history = st.tabs(["📡 Live Autonomous Radar", "📊 Historical Intelligence & Logs"])

# --- 1. OFFLINE DATABASES ---
@st.cache_data
def get_country_list():
    return sorted([country.name for country in pycountry.countries])

@st.cache_data
def get_airport_db():
    return airportsdata.load('IATA')

st.sidebar.header("Dashboard Settings")
region_choice = st.sidebar.selectbox("Select Airspace Region:", get_country_list())

# --- 2. AIRLABS AIRLINE DATABASE ---
@st.cache_data(ttl=86400)
def get_airlines_db():
    api_key = st.secrets.get("AIRLABS_API_KEY", None)
    if not api_key: return {}
    url = f"https://airlabs.co/api/v9/airlines?api_key={api_key}"
    try:
        res = requests.get(url, timeout=10).json()
        airlines = {}
        for a in res.get('response', []):
            if a.get('icao_code'): airlines[a['icao_code']] = a.get('name')
            if a.get('iata_code'): airlines[a['iata_code']] = a.get('name')
        return airlines
    except:
        return {}

# --- 3. BOUNDING BOX ---
@st.cache_data
def get_bounding_box(country_name):
    headers = {"User-Agent": "StreamlitFlightTracker/1.0"}
    url = f"https://nominatim.openstreetmap.org/search?country={country_name}&format=json"
    try:
        res = requests.get(url, headers=headers, timeout=5).json()
        if res:
            bbox = res[0]['boundingbox'] 
            return {
                "south": float(bbox[0]), "north": float(bbox[1]),
                "west": float(bbox[2]), "east": float(bbox[3]),
                "airlabs_format": f"{bbox[0]},{bbox[2]},{bbox[1]},{bbox[3]}"
            }
    except:
        pass
    return None

# --- 4. FETCH DATA ---
@st.cache_data(ttl=45) 
def fetch_flight_data(bbox):
    if not bbox: return [], "none"
    api_key = st.secrets.get("AIRLABS_API_KEY", None)
    
    if api_key:
        url = f"https://airlabs.co/api/v9/flights?api_key={api_key}&bbox={bbox['airlabs_format']}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json().get('response', [])
                if data: return data, "airlabs"
        except:
            pass

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

# --- 5. AIRCRAFT PARSER ---
def parse_aircraft(icao_code):
    if not icao_code: return "Unknown", "Unknown Model"
    icao_code = str(icao_code).upper()
    if icao_code.startswith('B7'): return "Boeing", icao_code
    if icao_code.startswith('A3'): return "Airbus", icao_code
    if icao_code.startswith('E1') or icao_code.startswith('E7'): return "Embraer", icao_code
    if icao_code.startswith('CRJ'): return "Bombardier", icao_code
    return "Various/Other", icao_code

# --- 6. ADVANCED PHYSICS & AI ENGINES ---
def calculate_physics_metrics(alt, speed, wind_speed):
    # Turbulence Index based on speed differential against live wind shear
    wind_factor = wind_speed if wind_speed else 10.0
    turbulence_score = min(100, max(5, (speed / 50.0) * (wind_factor / 15.0)))
    
    # Estimated Carbon Emissions Index (kg/sec proxy based on altitude and speed load)
    carbon_rate = round((speed * 0.4) + (max(0, 10000 - alt) * 0.01), 2)
    
    if turbulence_score > 75: turb_label = "🔴 Severe Turbulence Risk"
    elif turbulence_score > 40: turb_label = "🟡 Moderate Airframe Stress"
    else: turb_label = "🟢 Smooth Airflow"
    
    return turb_label, carbon_rate

def generate_anomaly_reason(alt, speed, v_speed, squawk):
    reasons = []
    if squawk in ['7700', '7500', '7600']:
        emergency_map = {'7700': '🚨 SQUAWK 7700: GENERAL EMERGENCY', '7500': '🏴‍☠️ SQUAWK 7500: UNLAWFUL INTERFERENCE / HIJACK', '7600': '📻 SQUAWK 7600: RADIO COMMUNICATION FAILURE'}
        reasons.append(emergency_map.get(squawk, f"🚨 EMERGENCY SQUAWK: {squawk}"))
        
    if v_speed < -15: reasons.append(f"Rapid Descent ({v_speed:.1f} m/s)")
    elif v_speed > 15: reasons.append(f"Steep Climb ({v_speed:.1f} m/s)")
    if alt > 3000 and speed < 100: reasons.append(f"Low Speed at Altitude ({speed:.0f} km/h)")
    elif alt < 1000 and speed > 550: reasons.append(f"Excessive Low-Altitude Speed ({speed:.0f} km/h)")
    
    if not reasons: reasons.append("Multi-Variable Trajectory Outlier")
    return " | ".join(reasons)

def get_flight_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,precipitation"
    try:
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            current = res.json().get('current', {})
            return current.get("temperature_2m", 0.0), current.get("wind_speed_10m", 0.0), current.get("precipitation", 0.0)
    except:
        pass
    return 0.0, 0.0, 0.0

# --- 7. WEBHOOK DISPATCHER ---
def trigger_webhook_alert(callsign, reason, region, lat, lon):
    webhook_url = st.secrets.get("WEBHOOK_URL", None)
    if not webhook_url: return
    
    payload = {
        "content": f"🚨 **AUTONOMOUS AVIATION ALERT** 🚨\n**Region:** {region}\n**Flight:** {callsign}\n**Trigger:** {reason}\n**GPS:** {lat}, {lon}"
    }
    try:
        requests.post(webhook_url, json=payload, timeout=3)
    except:
        pass

# --- 8. TRANSFORM & ML PIPELINE ---
def process_data(raw_data, source_type):
    if not raw_data: return pd.DataFrame()
        
    airports_db = get_airport_db()
    airlines_db = get_airlines_db()
    clean_list = []
    
    if source_type == "airlabs":
        for item in raw_data:
            dep_iata = item.get('dep_iata', '')
            arr_iata = item.get('arr_iata', '')
            dep_info = airports_db.get(dep_iata, {})
            arr_info = airports_db.get(arr_iata, {})
            
            dep_full = f"{dep_info.get('name', dep_iata)} ({dep_info.get('city', 'Unknown')})" if dep_iata else "Unknown Airport"
            arr_full = f"{arr_info.get('name', arr_iata)} ({arr_info.get('city', 'Unknown')})" if arr_iata else "Unknown Airport"
            
            airline_code = item.get('airline_icao', item.get('airline_iata', ''))
            airline_name = airlines_db.get(airline_code, "Private/Unknown Carrier")
            manufacturer, model = parse_aircraft(item.get('aircraft_icao', ''))

            clean_list.append({
                'callsign': str(item.get('flight_icao', item.get('reg_number', 'UNKNOWN'))).strip(),
                'airline_name': airline_name,
                'manufacturer': manufacturer,
                'aircraft_model': model,
                'dep_full': dep_full,
                'arr_full': arr_full,
                'lat': item.get('lat'),
                'lon': item.get('lng'),
                'alt': item.get('alt', 0),
                'speed': item.get('speed', 0),
                'v_speed': item.get('v_speed', 0),
                'squawk': str(item.get('squawk', ''))
            })
    elif source_type == "opensky":
        for state in raw_data:
            clean_list.append({
                'callsign': str(state[1]).strip() if state[1] else "UNKNOWN",
                'airline_name': "Unavailable (OpenSky)",
                'manufacturer': "Unknown", 'aircraft_model': "Unknown",
                'dep_full': "Unavailable", 'arr_full': "Unavailable",
                'lat': state[6], 'lon': state[5],
                'alt': state[7] if state[7] is not None else 0,
                'speed': state[9] if state[9] is not None else 0,
                'v_speed': state[11] if state[11] is not None else 0,
                'squawk': str(state[14]) if len(state) > 14 and state[14] else ''
            })
            
    df = pd.DataFrame(clean_list)
    df = df.dropna(subset=['lat', 'lon'])
    
    if len(df) > 5:
        features = df[['alt', 'speed', 'v_speed']]
        model = IsolationForest(contamination=0.03, random_state=42)
        df['anomaly_score'] = model.fit_predict(features)
        
        # Force flag emergency squawks regardless of ML model score
        df['is_emergency'] = df['squawk'].isin(['7700', '7500', '7600'])
        df['status'] = df.apply(lambda x: '🚨 EMERGENCY' if x['is_emergency'] else ('⚠️ Anomaly' if x['anomaly_score'] == -1 else '✅ Normal'), axis=1)
        df['color'] = df['status'].apply(lambda x: '#8b0000' if x == '🚨 EMERGENCY' else ('#ff0000' if x == '⚠️ Anomaly' else '#0000ff'))
        
        reasons, temps, winds, precips, turb_list, carbon_list = [], [], [], [], [], []
        
        for _, row in df.iterrows():
            if row['status'] != '✅ Normal':
                reason = generate_anomaly_reason(row['alt'], row['speed'], row['v_speed'], row['squawk'])
                t, w, p = get_flight_weather(row['lat'], row['lon'])
                turb, carb = calculate_physics_metrics(row['alt'], row['speed'], w)
                
                reasons.append(reason); temps.append(t); winds.append(w); precips.append(p)
                turb_list.append(turb); carbon_list.append(carb)
            else:
                reasons.append("Normal"); temps.append(None); winds.append(None); precips.append(None)
                turb_list.append("Normal"); carbon_list.append(0.0)
                
        df['anomaly_reason'] = reasons
        df['temp_c'] = temps; df['wind_speed_kmh'] = winds; df['precipitation_mm'] = precips
        df['turbulence_index'] = turb_list; df['carbon_rate'] = carbon_list
    else:
        df['status'] = '✅ Normal'; df['color'] = '#0000ff'
        df['anomaly_reason'] = 'Normal'
        df['temp_c'], df['wind_speed_kmh'], df['precipitation_mm'] = None, None, None
        df['turbulence_index'], df['carbon_rate'] = "Normal", 0.0
        
    return df

# --- 9. DATABASE LOGGING ---
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
            reason_full = f"[{region}] {row['anomaly_reason']} | Carbon: {row.get('carbon_rate')}kg/s"
            cursor.execute(insert_query, (
                row['callsign'] if row['callsign'] else "UNKNOWN",
                float(row.get('alt', 0)), float(row.get('speed', 0)), float(row.get('v_speed', 0)), reason_full
            ))
            # Trigger autonomous webhook dispatch
            trigger_webhook_alert(row['callsign'], row['anomaly_reason'], region, row['lat'], row['lon'])
            
        conn.commit()
        cursor.close(); conn.close()
    except:
        pass

def fetch_historical_data():
    db_uri = st.secrets.get("DB_URI", None)
    if not db_uri: return pd.DataFrame()
    try:
        conn = psycopg2.connect(db_uri)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, callsign, baro_altitude, velocity, vertical_rate, anomaly_reason FROM historical_anomalies ORDER BY timestamp DESC LIMIT 500")
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=columns)
        cursor.close(); conn.close()
        return df
    except:
        return pd.DataFrame()

# ==========================================
# DASHBOARD INTERFACE
# ==========================================
with tab_live:
    bbox = get_bounding_box(region_choice)
    with st.spinner(f'Executing autonomous scan over {region_choice}...'):
        raw_flights, source_type = fetch_flight_data(bbox)
        
    df = process_data(raw_flights, source_type)
    
    if df.empty:
        st.warning(f"No active flight data currently retrieved for {region_choice}.")
    else:
        anomalies = df[df['status'] != '✅ Normal']
        log_anomalies_to_sql(anomalies, region_choice)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Autonomous Radar Matrix: {len(df)} Active Flights")
            st.map(df, latitude='lat', longitude='lon', color='color', zoom=4)
            
        with col2:
            st.subheader(f"Flagged Events: {len(anomalies)}")
            if not anomalies.empty:
                for _, row in anomalies.iterrows():
                    with st.expander(f"⚠️ {row['callsign']} ({row['airline_name']})"):
                        st.error(f"**Analysis:** {row['anomaly_reason']}")
                        
                        st.write("---")
                        st.markdown(f"**🏢 Airline:** {row['airline_name']}")
                        st.markdown(f"**🛠️ Aircraft:** {row['manufacturer']} ({row['aircraft_model']})")
                        st.markdown(f"**🛫 From:** {row['dep_full']}")
                        st.markdown(f"**🛬 To:** {row['arr_full']}")
                        
                        st.write("---")
                        st.write("**🧠 Physics & Environmental AI**")
                        st.info(f"**Airframe Stress:** {row['turbulence_index']}")
                        st.metric("Est. Carbon Burn Rate", f"{row['carbon_rate']} kg/s")
                        
                        st.write("---")
                        st.write("**📡 Telemetry & Weather**")
                        m1, m2 = st.columns(2)
                        m1.metric("Altitude", f"{row['alt']} m")
                        m2.metric("Velocity", f"{row['speed']} km/h")
                        w1, w2 = st.columns(2)
                        w1.metric("Temp", f"{row['temp_c']} °C")
                        w2.metric("Wind", f"{row['wind_speed_kmh']} km/h")
            else:
                st.success("Airspace behavior is normal and stable.")

with tab_history:
    st.subheader("Autonomous Event Database & Trends")
    hist_df = fetch_historical_data()
    
    if not hist_df.empty:
        st.markdown("### 1. Velocity vs. Altitude of Logged Events")
        st.scatter_chart(hist_df, x='velocity', y='baro_altitude', color='#ff0000')
        st.markdown("### 2. Live Supabase Event Logs")
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No events logged yet.")
