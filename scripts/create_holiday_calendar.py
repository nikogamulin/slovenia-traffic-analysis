#!/usr/bin/env python3
"""
Create comprehensive holiday calendar for Slovenia, Germany, Austria, and Italy
Includes public holidays and school holidays for transit traffic analysis
"""

import pandas as pd
from datetime import datetime, timedelta

def create_slovenia_holidays():
    """Create Slovenia public and school holidays"""
    holidays = []
    
    for year in range(2020, 2026):
        # Fixed public holidays
        holidays.extend([
            {'date': f'{year}-01-01', 'country': 'SI', 'holiday_name': "New Year's Day", 'type': 'public'},
            {'date': f'{year}-01-02', 'country': 'SI', 'holiday_name': "New Year Holiday", 'type': 'public'},
            {'date': f'{year}-02-08', 'country': 'SI', 'holiday_name': "Prešeren Day", 'type': 'public'},
            {'date': f'{year}-04-27', 'country': 'SI', 'holiday_name': "Day of Uprising", 'type': 'public'},
            {'date': f'{year}-05-01', 'country': 'SI', 'holiday_name': "Labour Day", 'type': 'public'},
            {'date': f'{year}-05-02', 'country': 'SI', 'holiday_name': "Labour Day Holiday", 'type': 'public'},
            {'date': f'{year}-06-25', 'country': 'SI', 'holiday_name': "Statehood Day", 'type': 'public'},
            {'date': f'{year}-08-15', 'country': 'SI', 'holiday_name': "Assumption of Mary", 'type': 'public'},
            {'date': f'{year}-10-31', 'country': 'SI', 'holiday_name': "Reformation Day", 'type': 'public'},
            {'date': f'{year}-11-01', 'country': 'SI', 'holiday_name': "All Saints' Day", 'type': 'public'},
            {'date': f'{year}-12-25', 'country': 'SI', 'holiday_name': "Christmas", 'type': 'public'},
            {'date': f'{year}-12-26', 'country': 'SI', 'holiday_name': "Independence Day", 'type': 'public'},
        ])
        
        # Easter Monday (moveable)
        easter_dates = {
            2020: '2020-04-13', 2021: '2021-04-05', 2022: '2022-04-18',
            2023: '2023-04-10', 2024: '2024-04-01', 2025: '2025-04-21'
        }
        if year in easter_dates:
            holidays.append({'date': easter_dates[year], 'country': 'SI', 
                           'holiday_name': 'Easter Monday', 'type': 'public'})
        
        # School holidays (approximate)
        # Winter holidays - two groups with different weeks
        holidays.extend([
            {'date': f'{year}-02-15', 'country': 'SI', 'holiday_name': 'Winter Holiday Group 1', 
             'type': 'school', 'region': 'Eastern Slovenia'},
            {'date': f'{year}-02-22', 'country': 'SI', 'holiday_name': 'Winter Holiday Group 2', 
             'type': 'school', 'region': 'Western Slovenia'},
        ])
        
        # Summer holidays
        for day in range(1, 32):  # July
            holidays.append({'date': f'{year}-07-{day:02d}', 'country': 'SI', 
                           'holiday_name': 'Summer Holiday', 'type': 'school'})
        for day in range(1, 32):  # August
            holidays.append({'date': f'{year}-08-{day:02d}', 'country': 'SI', 
                           'holiday_name': 'Summer Holiday', 'type': 'school'})
    
    return holidays

def create_germany_holidays():
    """Create Germany public and school holidays (focus on Bavaria and Baden-Württemberg)"""
    holidays = []
    
    for year in range(2020, 2026):
        # National public holidays
        holidays.extend([
            {'date': f'{year}-01-01', 'country': 'DE', 'holiday_name': "New Year's Day", 'type': 'public'},
            {'date': f'{year}-05-01', 'country': 'DE', 'holiday_name': "Labour Day", 'type': 'public'},
            {'date': f'{year}-10-03', 'country': 'DE', 'holiday_name': "German Unity Day", 'type': 'public'},
            {'date': f'{year}-12-25', 'country': 'DE', 'holiday_name': "Christmas Day", 'type': 'public'},
            {'date': f'{year}-12-26', 'country': 'DE', 'holiday_name': "Boxing Day", 'type': 'public'},
        ])
        
        # Bavaria-specific holidays
        holidays.extend([
            {'date': f'{year}-01-06', 'country': 'DE', 'holiday_name': "Epiphany", 
             'type': 'public', 'region': 'Bavaria'},
            {'date': f'{year}-08-15', 'country': 'DE', 'holiday_name': "Assumption of Mary", 
             'type': 'public', 'region': 'Bavaria'},
            {'date': f'{year}-11-01', 'country': 'DE', 'holiday_name': "All Saints' Day", 
             'type': 'public', 'region': 'Bavaria'},
        ])
        
        # Good Friday and Easter Monday (moveable)
        easter_dates = {
            2020: ('2020-04-10', '2020-04-13'),
            2021: ('2021-04-02', '2021-04-05'),
            2022: ('2022-04-15', '2022-04-18'),
            2023: ('2023-04-07', '2023-04-10'),
            2024: ('2024-03-29', '2024-04-01'),
            2025: ('2025-04-18', '2025-04-21')
        }
        if year in easter_dates:
            holidays.append({'date': easter_dates[year][0], 'country': 'DE', 
                           'holiday_name': 'Good Friday', 'type': 'public'})
            holidays.append({'date': easter_dates[year][1], 'country': 'DE', 
                           'holiday_name': 'Easter Monday', 'type': 'public'})
        
        # Summer school holidays (Bavaria/Baden-Württemberg typically late July-early September)
        # Staggered by state
        if year == 2020:
            start_date = datetime(2020, 7, 27)
        elif year == 2021:
            start_date = datetime(2021, 7, 30)
        elif year == 2022:
            start_date = datetime(2022, 8, 1)
        elif year == 2023:
            start_date = datetime(2023, 7, 31)
        elif year == 2024:
            start_date = datetime(2024, 7, 29)
        else:  # 2025
            start_date = datetime(2025, 7, 28)
        
        for i in range(42):  # 6 weeks summer holiday
            holiday_date = start_date + timedelta(days=i)
            holidays.append({
                'date': holiday_date.strftime('%Y-%m-%d'),
                'country': 'DE',
                'holiday_name': 'Summer Holiday',
                'type': 'school',
                'region': 'Bavaria/Baden-Württemberg'
            })
    
    return holidays

def create_austria_holidays():
    """Create Austria public and school holidays"""
    holidays = []
    
    for year in range(2020, 2026):
        # Public holidays
        holidays.extend([
            {'date': f'{year}-01-01', 'country': 'AT', 'holiday_name': "New Year's Day", 'type': 'public'},
            {'date': f'{year}-01-06', 'country': 'AT', 'holiday_name': "Epiphany", 'type': 'public'},
            {'date': f'{year}-05-01', 'country': 'AT', 'holiday_name': "Labour Day", 'type': 'public'},
            {'date': f'{year}-08-15', 'country': 'AT', 'holiday_name': "Assumption of Mary", 'type': 'public'},
            {'date': f'{year}-10-26', 'country': 'AT', 'holiday_name': "National Day", 'type': 'public'},
            {'date': f'{year}-11-01', 'country': 'AT', 'holiday_name': "All Saints' Day", 'type': 'public'},
            {'date': f'{year}-12-08', 'country': 'AT', 'holiday_name': "Immaculate Conception", 'type': 'public'},
            {'date': f'{year}-12-25', 'country': 'AT', 'holiday_name': "Christmas Day", 'type': 'public'},
            {'date': f'{year}-12-26', 'country': 'AT', 'holiday_name': "St. Stephen's Day", 'type': 'public'},
        ])
        
        # Easter Monday and other moveable holidays
        easter_dates = {
            2020: ('2020-04-13', '2020-05-21', '2020-06-01', '2020-06-11'),  # Easter Mon, Ascension, Whit Mon, Corpus Christi
            2021: ('2021-04-05', '2021-05-13', '2021-05-24', '2021-06-03'),
            2022: ('2022-04-18', '2022-05-26', '2022-06-06', '2022-06-16'),
            2023: ('2023-04-10', '2023-05-18', '2023-05-29', '2023-06-08'),
            2024: ('2024-04-01', '2024-05-09', '2024-05-20', '2024-05-30'),
            2025: ('2025-04-21', '2025-05-29', '2025-06-09', '2025-06-19')
        }
        if year in easter_dates:
            holidays.extend([
                {'date': easter_dates[year][0], 'country': 'AT', 'holiday_name': 'Easter Monday', 'type': 'public'},
                {'date': easter_dates[year][1], 'country': 'AT', 'holiday_name': 'Ascension Day', 'type': 'public'},
                {'date': easter_dates[year][2], 'country': 'AT', 'holiday_name': 'Whit Monday', 'type': 'public'},
                {'date': easter_dates[year][3], 'country': 'AT', 'holiday_name': 'Corpus Christi', 'type': 'public'},
            ])
        
        # Summer school holidays (typically early July to early September)
        start_date = datetime(year, 7, 1)
        for i in range(63):  # 9 weeks summer holiday
            holiday_date = start_date + timedelta(days=i)
            holidays.append({
                'date': holiday_date.strftime('%Y-%m-%d'),
                'country': 'AT',
                'holiday_name': 'Summer Holiday',
                'type': 'school'
            })
    
    return holidays

def create_italy_holidays():
    """Create Italy public and school holidays (focus on northern regions)"""
    holidays = []
    
    for year in range(2020, 2026):
        # National public holidays
        holidays.extend([
            {'date': f'{year}-01-01', 'country': 'IT', 'holiday_name': "New Year's Day", 'type': 'public'},
            {'date': f'{year}-01-06', 'country': 'IT', 'holiday_name': "Epiphany", 'type': 'public'},
            {'date': f'{year}-04-25', 'country': 'IT', 'holiday_name': "Liberation Day", 'type': 'public'},
            {'date': f'{year}-05-01', 'country': 'IT', 'holiday_name': "Labour Day", 'type': 'public'},
            {'date': f'{year}-06-02', 'country': 'IT', 'holiday_name': "Republic Day", 'type': 'public'},
            {'date': f'{year}-08-15', 'country': 'IT', 'holiday_name': "Ferragosto", 'type': 'public'},
            {'date': f'{year}-11-01', 'country': 'IT', 'holiday_name': "All Saints' Day", 'type': 'public'},
            {'date': f'{year}-12-08', 'country': 'IT', 'holiday_name': "Immaculate Conception", 'type': 'public'},
            {'date': f'{year}-12-25', 'country': 'IT', 'holiday_name': "Christmas Day", 'type': 'public'},
            {'date': f'{year}-12-26', 'country': 'IT', 'holiday_name': "St. Stephen's Day", 'type': 'public'},
        ])
        
        # Easter Monday (moveable)
        easter_dates = {
            2020: '2020-04-13', 2021: '2021-04-05', 2022: '2022-04-18',
            2023: '2023-04-10', 2024: '2024-04-01', 2025: '2025-04-21'
        }
        if year in easter_dates:
            holidays.append({'date': easter_dates[year], 'country': 'IT', 
                           'holiday_name': 'Easter Monday', 'type': 'public'})
        
        # Summer school holidays (mid-June to mid-September - longest in Europe)
        start_date = datetime(year, 6, 10)
        for i in range(95):  # ~13-14 weeks summer holiday
            holiday_date = start_date + timedelta(days=i)
            holidays.append({
                'date': holiday_date.strftime('%Y-%m-%d'),
                'country': 'IT',
                'holiday_name': 'Summer Holiday',
                'type': 'school',
                'region': 'Northern Italy'
            })
        
        # Christmas holidays
        for day in range(23, 32):
            holidays.append({'date': f'{year}-12-{day:02d}', 'country': 'IT', 
                           'holiday_name': 'Christmas Holiday', 'type': 'school'})
        for day in range(1, 7):
            holidays.append({'date': f'{year+1}-01-{day:02d}', 'country': 'IT', 
                           'holiday_name': 'Christmas Holiday', 'type': 'school'})
    
    return holidays

def create_combined_holiday_calendar():
    """Combine all country holidays into single DataFrame"""
    
    print("Creating holiday calendars for 4 countries...")
    
    # Collect all holidays
    all_holidays = []
    all_holidays.extend(create_slovenia_holidays())
    all_holidays.extend(create_germany_holidays())
    all_holidays.extend(create_austria_holidays())
    all_holidays.extend(create_italy_holidays())
    
    # Convert to DataFrame
    df = pd.DataFrame(all_holidays)
    
    # Add region column where missing
    df['region'] = df.get('region', 'National')
    
    # Sort by date and country
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'country'])
    
    # Add day of week
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Add year and month columns for easier filtering
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    return df

if __name__ == "__main__":
    print("Creating comprehensive holiday calendar (2020-2025)")
    print("=" * 50)
    
    # Create combined calendar
    holiday_df = create_combined_holiday_calendar()
    
    # Save to CSV
    output_file = 'data/external/holidays/holidays_combined_2020_2025.csv'
    holiday_df.to_csv(output_file, index=False)
    print(f"Saved {len(holiday_df)} holiday records to {output_file}")
    
    # Summary statistics
    print("\nHoliday Summary by Country:")
    summary = holiday_df.groupby(['country', 'type']).size().unstack(fill_value=0)
    print(summary)
    
    print("\nSample holidays (August 2023 - peak transit period):")
    august_2023 = holiday_df[(holiday_df['year'] == 2023) & (holiday_df['month'] == 8) & 
                             (holiday_df['type'] == 'public')]
    print(august_2023[['date', 'country', 'holiday_name']])
    
    print("\nSchool holiday periods by country (2025):")
    school_2025 = holiday_df[(holiday_df['year'] == 2025) & (holiday_df['type'] == 'school')]
    for country in ['SI', 'DE', 'AT', 'IT']:
        country_schools = school_2025[school_2025['country'] == country]
        if len(country_schools) > 0:
            print(f"{country}: {country_schools['date'].min().date()} to {country_schools['date'].max().date()}")
    
    print("\nHoliday calendar created successfully!")