# flight-anomaly-tracker

# Global Aviation Radar & Anomaly Tracker

A real-time aviation tracking and anomaly detection dashboard built with Streamlit, Python, and Supabase PostgreSQL.

## Features
* **Live Global Airspace Tracking:** Select any country to track active flights.
* **ML Anomaly Detection:** Uses Isolation Forest to flag unusual flight trajectories.
* **Weather Enrichment:** Integrates live weather data from Open-Meteo for flagged coordinates.
* **Database Logging:** Automatically saves detected anomalies to a Supabase PostgreSQL database.

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
   cd YOUR_REPOSITORY


   pip install -r requirements.txt

   streamlit run app.py
