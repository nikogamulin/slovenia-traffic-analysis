#!/usr/bin/env python3
"""
Merge Production Data Script
Processes all CSV files from production_data folder and creates merged CSVs
with road names and direction information.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import sys
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Road mapping dictionary
ROAD_MAPPING = {
    '0011': {
        'name': 'Bertoki HC',
        'direction_A': 'Bertoki → Koper',
        'direction_B': 'Koper → Bertoki'
    },
    '0015a': {
        'name': 'Maribor HC',
        'direction_A': 'Center → Pesnica',
        'direction_B': 'Pesnica → Center'
    },
    '0015b': {
        'name': 'Maribor HC',
        'direction_A': 'Center → Slivnica',
        'direction_B': 'Slivnica → Center'
    },
    '0016a': {
        'name': 'Maliska HC',
        'direction_A': 'Maliska → Center',
        'direction_B': 'Center → Maliska'
    },
    '0021': {
        'name': 'Ljubljana Ring',
        'direction_A': 'Clockwise',
        'direction_B': 'Counter-clockwise'
    },
    '0031': {
        'name': 'Koper-Ljubljana',
        'direction_A': 'Koper → Ljubljana',
        'direction_B': 'Ljubljana → Koper'
    },
    '0041': {
        'name': 'Celje-Maribor',
        'direction_A': 'Celje → Maribor',
        'direction_B': 'Maribor → Celje'
    },
    '0051': {
        'name': 'Ljubljana-Celje',
        'direction_A': 'Ljubljana → Celje',
        'direction_B': 'Celje → Ljubljana'
    },
    '0061': {
        'name': 'Maribor-Ptuj',
        'direction_A': 'Maribor → Ptuj',
        'direction_B': 'Ptuj → Maribor'
    },
    '0071': {
        'name': 'Ljubljana-Kranj',
        'direction_A': 'Ljubljana → Kranj',
        'direction_B': 'Kranj → Ljubljana'
    },
    '0081': {
        'name': 'Celje-Velenje',
        'direction_A': 'Celje → Velenje',
        'direction_B': 'Velenje → Celje'
    },
    '0091': {
        'name': 'Novo Mesto-Ljubljana',
        'direction_A': 'Novo Mesto → Ljubljana',
        'direction_B': 'Ljubljana → Novo Mesto'
    },
    '0101': {
        'name': 'Postojna-Koper',
        'direction_A': 'Postojna → Koper',
        'direction_B': 'Koper → Postojna'
    },
    '0111': {
        'name': 'Ljubljana-Novo Mesto',
        'direction_A': 'Ljubljana → Novo Mesto',
        'direction_B': 'Novo Mesto → Ljubljana'
    },
    '0121': {
        'name': 'Kranj-Bled',
        'direction_A': 'Kranj → Bled',
        'direction_B': 'Bled → Kranj'
    },
    '0131': {
        'name': 'Velenje-Maribor',
        'direction_A': 'Velenje → Maribor',
        'direction_B': 'Maribor → Velenje'
    },
    '0141': {
        'name': 'Murska Sobota HC',
        'direction_A': 'Center → Border',
        'direction_B': 'Border → Center'
    },
    '0151': {
        'name': 'Ljubljana Bypass',
        'direction_A': 'North → South',
        'direction_B': 'South → North'
    },
    '0161': {
        'name': 'Koper Port',
        'direction_A': 'Port → City',
        'direction_B': 'City → Port'
    },
    '0171': {
        'name': 'Bled-Austria Border',
        'direction_A': 'Bled → Austria',
        'direction_B': 'Austria → Bled'
    }
}


def extract_info_from_path(file_path):
    """Extract road code and date from file path."""
    parts = file_path.parts
    filename = file_path.stem  # e.g., '0041_20250817'
    
    # Extract road code from path or filename
    road_code = None
    for part in parts:
        if part in ROAD_MAPPING:
            road_code = part
            break
    
    # If not found in path, try filename
    if not road_code and '_' in filename:
        potential_code = filename.split('_')[0]
        if potential_code in ROAD_MAPPING:
            road_code = potential_code
    
    # Extract date from filename
    date_str = None
    if '_' in filename:
        date_part = filename.split('_')[-1]
        if len(date_part) == 8 and date_part.isdigit():
            try:
                date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                # Validate date
                datetime.strptime(date_str, '%Y-%m-%d')
            except:
                date_str = None
    
    return road_code, date_str


def process_vehicle_count_file(file_path):
    """Process a single vehicle count CSV file."""
    try:
        road_code, date_str = extract_info_from_path(file_path)
        
        if not road_code or not date_str:
            logger.warning(f"Could not extract info from {file_path}")
            return None
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Add metadata
        road_info = ROAD_MAPPING.get(road_code, {})
        df['road_code'] = road_code
        df['road_name'] = road_info.get('name', 'Unknown')
        df['date'] = date_str
        df['direction_A_name'] = road_info.get('direction_A', 'Direction A')
        df['direction_B_name'] = road_info.get('direction_B', 'Direction B')
        
        # Calculate direction counts based on lanes
        # Assuming Lane_1 is direction A, Lane_2 and Lane_3 are direction B
        if 'Lane_1' in df.columns:
            df['direction_A_count'] = df['Lane_1']
        else:
            df['direction_A_count'] = 0
            
        if 'Lane_2' in df.columns and 'Lane_3' in df.columns:
            df['direction_B_count'] = df['Lane_2'] + df['Lane_3']
        elif 'Lane_2' in df.columns:
            df['direction_B_count'] = df['Lane_2']
        else:
            df['direction_B_count'] = 0
        
        # Reorder columns to match expected format
        cols_order = ['road_name', 'road_code', 'date', 'Time', 
                     'direction_A_name', 'direction_B_name',
                     'direction_A_count', 'direction_B_count']
        
        # Add remaining columns
        for col in df.columns:
            if col not in cols_order:
                cols_order.append(col)
        
        # Reorder and return
        return df[cols_order]
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def process_vehicle_speed_file(file_path):
    """Process a single vehicle speed CSV file."""
    try:
        road_code, date_str = extract_info_from_path(file_path)
        
        if not road_code or not date_str:
            logger.warning(f"Could not extract info from {file_path}")
            return None
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Add metadata
        road_info = ROAD_MAPPING.get(road_code, {})
        df['road_code'] = road_code
        df['road_name'] = road_info.get('name', 'Unknown')
        df['date'] = date_str
        df['direction_A_name'] = road_info.get('direction_A', 'Direction A')
        df['direction_B_name'] = road_info.get('direction_B', 'Direction B')
        
        # Reorder columns to match expected format
        cols_order = ['road_name', 'road_code', 'date', 'Time',
                     'direction_A_name', 'direction_B_name']
        
        # Add remaining columns
        for col in df.columns:
            if col not in cols_order:
                cols_order.append(col)
        
        # Reorder and return
        return df[cols_order]
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def merge_csv_files(data_type='vehicle_count', sample_size=None):
    """
    Merge all CSV files of specified type.
    
    Args:
        data_type: 'vehicle_count' or 'vehicle_speed'
        sample_size: If set, only process this many files (for testing)
    """
    base_path = Path('/home/niko/workspace/slovenia-trafffic-v2/data/production_data')
    data_path = base_path / data_type
    
    # Find all CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    
    if sample_size:
        csv_files = csv_files[:sample_size]
    
    logger.info(f"Found {len(csv_files)} {data_type} CSV files to process")
    
    # Process files
    all_dfs = []
    processor = process_vehicle_count_file if data_type == 'vehicle_count' else process_vehicle_speed_file
    
    for file_path in tqdm(csv_files, desc=f"Processing {data_type} files"):
        df = processor(file_path)
        if df is not None:
            all_dfs.append(df)
    
    # Combine all dataframes
    if all_dfs:
        logger.info(f"Combining {len(all_dfs)} dataframes...")
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by road_code, date, and time
        merged_df = merged_df.sort_values(['road_code', 'date', 'Time'])
        
        # Create datetime column
        merged_df['datetime'] = pd.to_datetime(merged_df['date'] + ' ' + merged_df['Time'])
        
        return merged_df
    else:
        logger.warning("No dataframes to merge")
        return None


def main():
    """Main execution function."""
    logger.info("Starting production data merge process...")
    
    # Process vehicle count data
    logger.info("Processing vehicle count data...")
    count_df = merge_csv_files('vehicle_count')
    
    if count_df is not None:
        output_path = Path('/home/niko/workspace/slovenia-trafffic-v2/data/production_merged_vehicle_count.csv')
        count_df.to_csv(output_path, index=False)
        logger.info(f"Saved vehicle count data to {output_path}")
        logger.info(f"Shape: {count_df.shape}, Date range: {count_df['date'].min()} to {count_df['date'].max()}")
    
    # Process vehicle speed data
    logger.info("Processing vehicle speed data...")
    speed_df = merge_csv_files('vehicle_speed')
    
    if speed_df is not None:
        output_path = Path('/home/niko/workspace/slovenia-trafffic-v2/data/production_merged_vehicle_speed.csv')
        speed_df.to_csv(output_path, index=False)
        logger.info(f"Saved vehicle speed data to {output_path}")
        logger.info(f"Shape: {speed_df.shape}, Date range: {speed_df['date'].min()} to {speed_df['date'].max()}")
    
    logger.info("Production data merge complete!")


if __name__ == "__main__":
    # For testing, you can run with a sample first
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        logger.info("Running in test mode with 10 files...")
        test_df = merge_csv_files('vehicle_count', sample_size=10)
        if test_df is not None:
            print("\nSample of merged data:")
            print(test_df.head())
            print(f"\nShape: {test_df.shape}")
            print(f"Columns: {test_df.columns.tolist()}")
    else:
        main()