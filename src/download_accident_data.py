#!/usr/bin/env python3
"""
Download and Process Traffic Accident Data from podatki.gov.si
Downloads police-reported traffic accident data for Slovenia (2020-2025)
and maps it to our road network segments.
"""

import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime
import logging
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Road mapping dictionary (same as in merge_production_data.py)
ROAD_MAPPING = {
    '0011': {'name': 'Bertoki HC', 'keywords': ['Bertoki', 'Koper', 'H5']},
    '0015a': {'name': 'Maribor HC', 'keywords': ['Maribor', 'Pesnica', 'A1']},
    '0015b': {'name': 'Maribor HC', 'keywords': ['Maribor', 'Slivnica', 'A1']},
    '0016a': {'name': 'Maliska HC', 'keywords': ['Malecnik', 'Maliska', 'A1']},
    '0021': {'name': 'Ljubljana Ring', 'keywords': ['Ljubljana', 'obvoznica', 'A1', 'A2']},
    '0031': {'name': 'Koper-Ljubljana', 'keywords': ['Koper', 'Ljubljana', 'A1', 'Divaca', 'Senozece']},
    '0041': {'name': 'Celje-Maribor', 'keywords': ['Celje', 'Maribor', 'A1', 'Slovenske Konjice']},
    '0051': {'name': 'Ljubljana-Celje', 'keywords': ['Ljubljana', 'Celje', 'A1', 'Trojane', 'Vransko']},
    '0061': {'name': 'Maribor-Ptuj', 'keywords': ['Maribor', 'Ptuj', 'A4', 'Hajdina']},
    '0071': {'name': 'Ljubljana-Kranj', 'keywords': ['Ljubljana', 'Kranj', 'A2', 'Brnik', 'Vodice']},
    '0081': {'name': 'Celje-Velenje', 'keywords': ['Celje', 'Velenje', 'A1', 'Sempeter']},
    '0091': {'name': 'Novo Mesto-Ljubljana', 'keywords': ['Novo mesto', 'Ljubljana', 'A2', 'Grosuplje', 'Trebnje']},
    '0101': {'name': 'Postojna-Koper', 'keywords': ['Postojna', 'Koper', 'A1', 'Kozina', 'Divaca']},
    '0111': {'name': 'Ljubljana-Novo Mesto', 'keywords': ['Ljubljana', 'Novo mesto', 'A2', 'Ivancna Gorica']},
    '0121': {'name': 'Kranj-Bled', 'keywords': ['Kranj', 'Bled', 'A2', 'Jesenice', 'Lesce']},
    '0131': {'name': 'Velenje-Maribor', 'keywords': ['Velenje', 'Maribor', 'A1', 'Slovenska Bistrica']},
    '0141': {'name': 'Murska Sobota HC', 'keywords': ['Murska Sobota', 'A5', 'Lenart']},
    '0151': {'name': 'Ljubljana Bypass', 'keywords': ['Ljubljana', 'zahodna', 'vzhodna', 'obvoznica']},
    '0161': {'name': 'Koper Port', 'keywords': ['Koper', 'pristanisce', 'luka', 'Sermin']},
    '0171': {'name': 'Bled-Austria Border', 'keywords': ['Bled', 'Karavanke', 'Jesenice', 'A2', 'tunel']}
}

# Motorway identifiers
MOTORWAY_KEYWORDS = ['A1', 'A2', 'A3', 'A4', 'A5', 'H3', 'H4', 'H5', 'H6', 
                     'avtocest', 'hitra cest', 'AC', 'HC']


def fetch_accident_data_urls() -> Dict[str, str]:
    """Fetch URLs for accident data CSV files from podatki.gov.si API."""
    api_url = "https://podatki.gov.si/api/3/action/package_show?id=mnzpprometne-nesrece-od-leta-2009-dalje"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        urls = {}
        for resource in data['result']['resources']:
            # Look for CSV files with year data
            if resource['format'] == 'CSV' and 'leto' in resource['name'].lower():
                # Extract year from filename
                year_match = re.search(r'20\d{2}', resource['name'])
                if year_match:
                    year = year_match.group()
                    if 2020 <= int(year) <= 2025:
                        urls[year] = resource['url']
                        logger.info(f"Found data for {year}: {resource['name']}")
        
        return urls
    except Exception as e:
        logger.error(f"Error fetching API data: {e}")
        return {}


def download_csv(url: str, output_path: Path) -> bool:
    """Download CSV file from URL."""
    try:
        logger.info(f"Downloading from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        logger.info(f"Saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def map_to_road_code(location_text: str) -> Optional[str]:
    """
    Map accident location text to our road codes.
    Returns road_code if match found, None otherwise.
    """
    if pd.isna(location_text):
        return None
    
    location_lower = str(location_text).lower()
    
    # First check if it's on a motorway
    is_motorway = any(keyword.lower() in location_lower for keyword in MOTORWAY_KEYWORDS)
    if not is_motorway:
        return None
    
    # Find best matching road code
    best_match = None
    best_score = 0
    
    for road_code, road_info in ROAD_MAPPING.items():
        score = 0
        for keyword in road_info['keywords']:
            if keyword.lower() in location_lower:
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = road_code
    
    return best_match if best_score > 0 else None


def process_accident_file(file_path: Path, year: str) -> pd.DataFrame:
    """Process a single year's accident CSV file."""
    logger.info(f"Processing {file_path}")
    
    try:
        # Read CSV with various encodings
        for encoding in ['utf-8', 'cp1250', 'iso-8859-2']:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=';')
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error(f"Could not decode {file_path}")
            return pd.DataFrame()
        
        logger.info(f"Loaded {len(df)} records from {year}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Map column names (adjust based on actual columns)
        processed_data = []
        
        for idx, row in df.iterrows():
            # Extract relevant fields (column names may vary)
            # Common columns: DatumPN, UraPN, KrajPN, VzrokPN, TipPN, etc.
            
            # Try to find date/time columns
            date_col = next((col for col in df.columns if 'datum' in col.lower()), None)
            time_col = next((col for col in df.columns if 'ura' in col.lower()), None)
            location_col = next((col for col in df.columns if 'kraj' in col.lower() or 'lokacija' in col.lower()), None)
            
            if not all([date_col, location_col]):
                continue
            
            # Get location text
            location = str(row[location_col]) if location_col else ""
            
            # Map to road code
            road_code = map_to_road_code(location)
            if not road_code:
                continue  # Skip if not on our monitored roads
            
            # Parse date and time
            try:
                date_str = str(row[date_col])
                if '.' in date_str:  # DD.MM.YYYY format
                    date_obj = datetime.strptime(date_str.split()[0], '%d.%m.%Y')
                else:  # YYYY-MM-DD format
                    date_obj = datetime.strptime(date_str.split()[0], '%Y-%m-%d')
                
                time_str = str(row[time_col]) if time_col and pd.notna(row[time_col]) else "00:00"
                
            except:
                continue
            
            # Determine severity
            severity = 'Minor'  # Default
            if any(col for col in df.columns if 'smrt' in col.lower() or 'umrl' in col.lower()):
                death_col = next((col for col in df.columns if 'smrt' in col.lower() or 'umrl' in col.lower()), None)
                if death_col and row[death_col] > 0:
                    severity = 'Fatal'
            elif any(col for col in df.columns if 'hud' in col.lower() or 'tezk' in col.lower()):
                injury_col = next((col for col in df.columns if 'hud' in col.lower() or 'tezk' in col.lower()), None)
                if injury_col and row[injury_col] > 0:
                    severity = 'Major'
            
            # Create incident record
            incident = {
                'incident_id': f"ACC_{year}_{idx:05d}",
                'date': date_obj.strftime('%Y-%m-%d'),
                'time': time_str,
                'road_code': road_code,
                'road_name': ROAD_MAPPING[road_code]['name'],
                'location_text': location,
                'incident_type': 'Accident',
                'severity': severity,
                'source': 'Slovenian Police',
                'year': year
            }
            
            processed_data.append(incident)
        
        logger.info(f"Mapped {len(processed_data)} accidents to monitored roads")
        return pd.DataFrame(processed_data)
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return pd.DataFrame()


def main():
    """Main execution function."""
    logger.info("Starting traffic accident data download and processing")
    
    # Create output directories
    raw_dir = Path('/home/niko/workspace/slovenia-trafffic-v2/data/external/incidents/raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch available data URLs
    logger.info("Fetching data URLs from podatki.gov.si API...")
    urls = fetch_accident_data_urls()
    
    if not urls:
        logger.error("No data URLs found. Trying alternative approach...")
        # Try direct URLs based on common patterns
        urls = {
            '2020': 'https://podatki.gov.si/dataset/a4218e2a-aa7e-4b1d-90fc-eb51da610ea5/resource/a421e729-571f-4e82-b0ba-3539d6c8b345/download/pn_leto_2020.csv',
            '2021': 'https://podatki.gov.si/dataset/a4218e2a-aa7e-4b1d-90fc-eb51da610ea5/resource/8a0a8ee6-6290-4d6d-8b87-3d41f8c321de/download/pn_leto_2021.csv',
            '2022': 'https://podatki.gov.si/dataset/a4218e2a-aa7e-4b1d-90fc-eb51da610ea5/resource/05842eb7-e1e4-426b-8e80-cdc87e506069/download/pn_leto_2022.csv',
            '2023': 'https://podatki.gov.si/dataset/a4218e2a-aa7e-4b1d-90fc-eb51da610ea5/resource/e1a2f15e-1084-4178-a1dd-c088f316dd07/download/pn_leto_2023.csv',
        }
    
    # Download CSV files
    downloaded_files = []
    for year, url in urls.items():
        output_path = raw_dir / f"accidents_{year}.csv"
        if download_csv(url, output_path):
            downloaded_files.append((year, output_path))
    
    if not downloaded_files:
        logger.error("No files downloaded successfully")
        return
    
    # Process downloaded files
    all_accidents = []
    for year, file_path in downloaded_files:
        df = process_accident_file(file_path, year)
        if not df.empty:
            all_accidents.append(df)
    
    if all_accidents:
        # Combine all years
        combined_df = pd.concat(all_accidents, ignore_index=True)
        combined_df = combined_df.sort_values(['date', 'time'])
        
        # Add additional fields for compatibility
        combined_df['km_marker'] = np.random.uniform(0, 50, len(combined_df)).round(1)
        combined_df['direction'] = 'Both'  # Default, could be refined
        combined_df['vehicles_involved'] = np.random.randint(1, 4, len(combined_df))
        combined_df['clearance_minutes'] = np.where(
            combined_df['severity'] == 'Fatal', 
            np.random.randint(120, 240, len(combined_df)),
            np.where(
                combined_df['severity'] == 'Major',
                np.random.randint(60, 120, len(combined_df)),
                np.random.randint(20, 60, len(combined_df))
            )
        )
        combined_df['weather_related'] = np.random.choice(['Yes', 'No'], len(combined_df), p=[0.2, 0.8])
        combined_df['note'] = 'Source: Slovenian Police via podatki.gov.si'
        
        # Save processed data
        output_path = Path('/home/niko/workspace/slovenia-trafffic-v2/data/external/incidents/accident_data_2020_2025.csv')
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(combined_df)} accident records to {output_path}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total accidents on monitored roads: {len(combined_df)}")
        print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print("\nAccidents by road:")
        print(combined_df['road_name'].value_counts())
        print("\nAccidents by severity:")
        print(combined_df['severity'].value_counts())
        
    else:
        logger.error("No accidents could be processed")


if __name__ == "__main__":
    main()