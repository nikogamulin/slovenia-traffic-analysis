#!/usr/bin/env python3
"""
Weather data collection script for Slovenia (ARSO)
This script collects historical weather data from ARSO weather stations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

# Key weather stations along Slovenian motorways
WEATHER_STATIONS = {
    'LJUBLJANA': {'id': 'LJUBL-ANA_BEZIGRAD', 'lat': 46.0658, 'lon': 14.5126},
    'MARIBOR': {'id': 'MARIBOR_SLIVNICA', 'lat': 46.4848, 'lon': 15.6866},
    'CELJE': {'id': 'CELJE_MEDLOG', 'lat': 46.2363, 'lon': 15.2266},
    'KOPER': {'id': 'PORTOROZ_LETALISCE', 'lat': 45.4749, 'lon': 13.6150},
    'KRANJ': {'id': 'BRNIK_LETALISCE', 'lat': 46.2237, 'lon': 14.4576},
    'NOVO_MESTO': {'id': 'NOVO_MESTO', 'lat': 45.8003, 'lon': 15.1773},
    'POSTOJNA': {'id': 'POSTOJNA', 'lat': 45.7682, 'lon': 14.1968},
    'MURSKA_SOBOTA': {'id': 'MURSKA_SOBOTA_RAKICAN', 'lat': 46.6525, 'lon': 16.1916}
}

def generate_sample_weather_data():
    """
    Generate sample weather data for demonstration
    In production, this would connect to ARSO API
    """
    
    # Date range: August 30, 2020 to August 29, 2025
    start_date = datetime(2020, 8, 30)
    end_date = datetime(2025, 8, 29)
    
    # Generate hourly timestamps
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    weather_data = []
    
    for station_name, station_info in WEATHER_STATIONS.items():
        print(f"Generating data for {station_name}...")
        
        for timestamp in date_range:
            # Seasonal variations
            month = timestamp.month
            hour = timestamp.hour
            
            # Base temperature by season
            if month in [12, 1, 2]:  # Winter
                base_temp = 2
            elif month in [3, 4, 5]:  # Spring
                base_temp = 15
            elif month in [6, 7, 8]:  # Summer
                base_temp = 25
            else:  # Autumn
                base_temp = 12
            
            # Daily temperature variation
            daily_var = 5 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else -2
            
            # Add random variation
            temp = base_temp + daily_var + np.random.normal(0, 3)
            
            # Precipitation probability (higher in autumn/spring)
            precip_prob = 0.3 if month in [3, 4, 5, 9, 10, 11] else 0.15
            precipitation = np.random.exponential(2) if np.random.random() < precip_prob else 0
            
            # Wind speed (higher in winter)
            base_wind = 15 if month in [11, 12, 1, 2] else 8
            wind_speed = max(0, base_wind + np.random.normal(0, 5))
            
            # Visibility (reduced during precipitation)
            visibility = 10000 if precipitation == 0 else max(1000, 10000 - precipitation * 1000)
            
            # Humidity
            humidity = min(100, 60 + precipitation * 5 + np.random.normal(0, 10))
            
            weather_data.append({
                'station_id': station_info['id'],
                'station_name': station_name,
                'latitude': station_info['lat'],
                'longitude': station_info['lon'],
                'datetime': timestamp,
                'temperature_c': round(temp, 1),
                'precipitation_mm': round(precipitation, 1),
                'wind_speed_kmh': round(wind_speed, 1),
                'wind_direction': np.random.randint(0, 360),
                'humidity_percent': round(humidity, 0),
                'pressure_hpa': round(1013 + np.random.normal(0, 10), 1),
                'visibility_m': round(visibility, 0),
                'weather_condition': 'Rain' if precipitation > 0 else 'Clear'
            })
    
    return pd.DataFrame(weather_data)

def save_weather_data(df, output_path):
    """Save weather data to CSV"""
    df.to_csv(output_path, index=False)
    print(f"Weather data saved to {output_path}")
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Stations: {df['station_name'].unique()}")

if __name__ == "__main__":
    print("Collecting Slovenia weather data (ARSO)...")
    print("Note: Using simulated data for demonstration")
    print("In production, implement arso-scraper or ARSO API connection")
    print("-" * 50)
    
    # Generate sample data
    weather_df = generate_sample_weather_data()
    
    # Save to CSV
    output_file = "data/external/weather/arso_weather_2020_2025.csv"
    save_weather_data(weather_df, output_file)
    
    # Display sample
    print("\nSample data:")
    print(weather_df.head(10))
    
    # Summary statistics
    print("\nSummary statistics:")
    print(weather_df[['temperature_c', 'precipitation_mm', 'wind_speed_kmh']].describe())