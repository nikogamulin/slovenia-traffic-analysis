#!/usr/bin/env python3
"""
Create economic data files for Slovenia traffic analysis
Includes fuel prices, GDP, and Value of Time estimates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_fuel_prices_data():
    """
    Create historical fuel price data for Slovenia (2020-2025)
    Based on research showing:
    - Minimum petrol: €0.94/L (Nov 2020)
    - Maximum petrol: €1.76/L (Jun 2022)
    - Minimum diesel: €0.98/L (Feb 2020)
    - Maximum diesel: €1.86/L (May 2022)
    """
    
    # Generate monthly dates
    dates = pd.date_range(start='2020-08-01', end='2025-08-01', freq='MS')
    
    fuel_data = []
    
    for date in dates:
        year = date.year
        month = date.month
        
        # Simulate price patterns based on historical trends
        # Low prices in 2020 (COVID), spike in 2022 (Ukraine crisis), moderate after
        
        if year == 2020:
            # COVID period - low prices
            petrol_base = 0.95 + month * 0.01
            diesel_base = 0.98 + month * 0.01
        elif year == 2021:
            # Recovery period - gradual increase
            petrol_base = 1.10 + (month / 12) * 0.20
            diesel_base = 1.15 + (month / 12) * 0.25
        elif year == 2022:
            # Crisis period - spike in prices
            if month <= 6:
                petrol_base = 1.40 + (month / 6) * 0.36
                diesel_base = 1.50 + (month / 6) * 0.36
            else:
                petrol_base = 1.76 - ((month - 6) / 6) * 0.20
                diesel_base = 1.86 - ((month - 6) / 6) * 0.25
        elif year == 2023:
            # Stabilization
            petrol_base = 1.45 - (month / 12) * 0.10
            diesel_base = 1.50 - (month / 12) * 0.12
        elif year == 2024:
            # Moderate prices
            petrol_base = 1.35 + np.sin(month * np.pi / 6) * 0.05
            diesel_base = 1.38 + np.sin(month * np.pi / 6) * 0.05
        else:  # 2025
            petrol_base = 1.30 - (month / 12) * 0.07
            diesel_base = 1.35 - (month / 12) * 0.08
        
        # Add random variation
        petrol_price = round(petrol_base + np.random.normal(0, 0.02), 3)
        diesel_price = round(diesel_base + np.random.normal(0, 0.02), 3)
        
        fuel_data.append({
            'date': date,
            'year': year,
            'month': month,
            'petrol_95_eur_per_liter': petrol_price,
            'diesel_eur_per_liter': diesel_price,
            'lpg_eur_per_liter': round(petrol_price * 0.55, 3),  # LPG typically 55% of petrol
            'source': 'Simulated based on historical patterns'
        })
    
    return pd.DataFrame(fuel_data)

def create_economic_indicators():
    """
    Create economic indicators including GDP and inflation
    """
    
    years = list(range(2020, 2026))
    
    # Slovenia economic data (approximate values)
    economic_data = {
        'year': years,
        'gdp_billion_eur': [48.0, 52.2, 57.0, 60.3, 62.4, 64.1],  # Estimated GDP
        'gdp_per_capita_eur': [22800, 24800, 27100, 28700, 29700, 30500],
        'inflation_rate_percent': [0.1, 4.9, 8.8, 6.9, 3.7, 2.5],  # Annual inflation
        'unemployment_rate_percent': [5.0, 4.6, 4.2, 3.8, 3.5, 3.6],
        'average_wage_eur': [1856, 1970, 2087, 2238, 2350, 2420],
        'population_millions': [2.108, 2.109, 2.107, 2.110, 2.112, 2.115]
    }
    
    return pd.DataFrame(economic_data)

def create_value_of_time():
    """
    Create Value of Time (VoT) estimates for different journey purposes
    Based on EU meta-analysis, adjusted for inflation
    """
    
    # Base VoT values from 2010 (EUR/hour)
    base_vot_2010 = {
        'commuter': 5.64,
        'business': 17.14,
        'leisure': 4.86,
        'freight': 20.81
    }
    
    # Inflation adjustment factors (cumulative from 2010)
    inflation_factors = {
        2020: 1.15,
        2021: 1.21,
        2022: 1.31,
        2023: 1.40,
        2024: 1.45,
        2025: 1.49
    }
    
    vot_data = []
    
    for year, factor in inflation_factors.items():
        for purpose, base_value in base_vot_2010.items():
            vot_data.append({
                'year': year,
                'journey_purpose': purpose,
                'vot_eur_per_hour': round(base_value * factor, 2),
                'base_value_2010': base_value,
                'inflation_factor': factor,
                'source': 'EU meta-analysis, inflation adjusted'
            })
    
    return pd.DataFrame(vot_data)

def save_economic_data():
    """Save all economic data files"""
    
    # Create fuel prices data
    print("Creating fuel prices data...")
    fuel_df = create_fuel_prices_data()
    fuel_df.to_csv('data/external/economic/fuel_prices_2020_2025.csv', index=False)
    print(f"Saved {len(fuel_df)} monthly fuel price records")
    
    # Create economic indicators
    print("\nCreating economic indicators...")
    economic_df = create_economic_indicators()
    economic_df.to_csv('data/external/economic/economic_indicators.csv', index=False)
    print(f"Saved economic indicators for {len(economic_df)} years")
    
    # Create Value of Time data
    print("\nCreating Value of Time estimates...")
    vot_df = create_value_of_time()
    vot_df.to_csv('data/external/economic/value_of_time.csv', index=False)
    print(f"Saved {len(vot_df)} VoT estimates")
    
    return fuel_df, economic_df, vot_df

if __name__ == "__main__":
    print("Creating economic data for Slovenia traffic analysis")
    print("=" * 50)
    
    fuel_df, economic_df, vot_df = save_economic_data()
    
    print("\n" + "=" * 50)
    print("Sample fuel prices (2022 peak period):")
    print(fuel_df[(fuel_df['year'] == 2022) & (fuel_df['month'].isin([5, 6]))][
        ['date', 'petrol_95_eur_per_liter', 'diesel_eur_per_liter']])
    
    print("\nEconomic indicators summary:")
    print(economic_df[['year', 'gdp_per_capita_eur', 'inflation_rate_percent', 'average_wage_eur']])
    
    print("\nValue of Time (2025):")
    print(vot_df[vot_df['year'] == 2025][['journey_purpose', 'vot_eur_per_hour']])
    
    print("\nAll economic data files created successfully!")