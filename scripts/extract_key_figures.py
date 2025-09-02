#!/usr/bin/env python3
"""
Key Figure Extraction Script - Manual Execution Approach
Extracts critical figures by running specific notebook sections

Task 6.3: Figure Extraction Component
"""

import sys
import os
sys.path.append('/home/niko/workspace/slovenia-trafffic-v2')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times', 'serif'],
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class KeyFigureExtractor:
    """Extract key figures from critical analysis steps"""
    
    def __init__(self):
        self.figures_dir = Path('/home/niko/workspace/slovenia-trafffic-v2/reports/article/figures')
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.figure_count = 20  # Start after existing figures
        
    def load_data(self):
        """Load the main datasets"""
        print("Loading datasets...")
        
        # Load traffic data
        try:
            self.df_counts = pd.read_csv('/home/niko/workspace/slovenia-trafffic-v2/data/production_merged_vehicle_count.csv')
            self.df_speeds = pd.read_csv('/home/niko/workspace/slovenia-trafffic-v2/data/production_merged_vehicle_speed.csv')
            
            # Basic merge
            self.df = pd.merge(
                self.df_counts[['road_name', 'road_code', 'date', 'Time', 'Total_All_Lanes', 'direction_A_count', 'direction_B_count']],
                self.df_speeds[['road_code', 'date', 'Time', 'Avg_Speed']],
                on=['road_code', 'date', 'Time'],
                how='inner'
            )
            
            # Create datetime
            self.df['datetime'] = pd.to_datetime(self.df['date'] + ' ' + self.df['Time'] + ':00')
            
            print(f"  → Loaded {len(self.df)} traffic records")
            
        except Exception as e:
            print(f"  → Error loading data: {e}")
            return False
        
        return True
    
    def extract_data_exploration_figures(self):
        """Create key data exploration visualizations"""
        print("\nExtracting data exploration figures...")
        
        # Figure: Traffic volume distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self.df['Total_All_Lanes'].dropna(), bins=50, 
                edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Traffic Volume (vehicles/hour)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Hourly Traffic Volume\nSlovenia Highway Network (2020-2025)')
        ax.grid(True, alpha=0.3)
        
        self._save_figure('traffic_volume_distribution')
        plt.close()
        
        # Figure: Speed distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self.df['Avg_Speed'].dropna(), bins=50, 
                edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Average Speed (km/h)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Average Highway Speeds\nSlovenia Highway Network (2020-2025)')
        ax.grid(True, alpha=0.3)
        
        self._save_figure('speed_distribution')
        plt.close()
        
        # Figure: Traffic trends by road
        selected_roads = ['Ljubljana-Celje', 'Koper-Ljubljana', 'Celje-Maribor']
        colors = ['#2E7D32', '#1565C0', '#E65100']
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for road, color in zip(selected_roads, colors):
            road_data = self.df[self.df['road_name'] == road].copy()
            if len(road_data) > 0:
                # Resample to monthly averages
                monthly = road_data.set_index('datetime').resample('M')['Total_All_Lanes'].mean()
                ax.plot(monthly.index, monthly.values, 
                       label=road, linewidth=2.5, color=color, alpha=0.8)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Monthly Traffic Volume (vehicles/hour)')
        ax.set_title('Highway Traffic Trends by Major Route\nMonthly Averages (2020-2025)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self._save_figure('highway_traffic_trends')
        plt.close()
        
        print("  → Data exploration figures extracted")
    
    def extract_speed_density_figures(self):
        """Create speed-density relationship figures"""
        print("\nExtracting speed-density analysis figures...")
        
        # Calculate density
        self.df['density'] = self.df['Total_All_Lanes'] / self.df['Avg_Speed'].replace(0, np.nan)
        self.df = self.df[self.df['density'].notna()]
        self.df = self.df[self.df['density'] < 20]  # Remove outliers
        
        # Figure: Speed-density scatter plot (fundamental diagram)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sample for performance
        sample_size = min(10000, len(self.df))
        df_sample = self.df.sample(n=sample_size)
        
        scatter = ax.scatter(df_sample['density'], df_sample['Avg_Speed'], 
                           alpha=0.5, s=10, c=df_sample['Total_All_Lanes'], 
                           cmap='viridis')
        
        ax.set_xlabel('Traffic Density (vehicles/km)')
        ax.set_ylabel('Average Speed (km/h)')
        ax.set_title('Speed-Density Relationship (Fundamental Diagram)\nSlovenia Highway Network')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, label='Traffic Volume (veh/h)')
        
        self._save_figure('speed_density_relationship')
        plt.close()
        
        # Figure: Speed-density bins heatmap
        speed_bins = pd.cut(self.df['Avg_Speed'], bins=10)
        density_bins = pd.cut(self.df['density'], bins=10)
        
        heatmap_data = pd.crosstab(density_bins, speed_bins)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd',
                   cbar_kws={'label': 'Observation Count'}, ax=ax)
        ax.set_title('Traffic State Distribution\nSpeed-Density Combinations')
        ax.set_xlabel('Speed Bins (km/h)')
        ax.set_ylabel('Density Bins (vehicles/km)')
        
        self._save_figure('traffic_state_heatmap')
        plt.close()
        
        print("  → Speed-density figures extracted")
    
    def extract_temporal_analysis_figures(self):
        """Create temporal analysis figures"""
        print("\nExtracting temporal analysis figures...")
        
        # Add time features
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        self.df['month'] = self.df['datetime'].dt.month
        
        # Figure: Hourly traffic patterns
        hourly_patterns = self.df.groupby(['hour', 'road_name'])['Total_All_Lanes'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for road in ['Ljubljana-Celje', 'Koper-Ljubljana', 'Celje-Maribor']:
            road_hourly = hourly_patterns[hourly_patterns['road_name'] == road]
            if len(road_hourly) > 0:
                ax.plot(road_hourly['hour'], road_hourly['Total_All_Lanes'], 
                       marker='o', linewidth=2, label=road, alpha=0.8)
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Traffic Volume (vehicles/hour)')
        ax.set_title('Daily Traffic Patterns by Highway Route')
        ax.set_xticks(range(0, 24, 2))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self._save_figure('daily_traffic_patterns')
        plt.close()
        
        # Figure: Weekly patterns
        weekly_patterns = self.df.groupby(['day_of_week', 'road_name'])['Total_All_Lanes'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for road in ['Ljubljana-Celje', 'Koper-Ljubljana', 'Celje-Maribor']:
            road_weekly = weekly_patterns[weekly_patterns['road_name'] == road]
            if len(road_weekly) > 0:
                ax.plot(road_weekly['day_of_week'], road_weekly['Total_All_Lanes'], 
                       marker='s', linewidth=2, label=road, alpha=0.8)
        
        ax.set_xlabel('Day of Week (0=Monday)')
        ax.set_ylabel('Average Traffic Volume (vehicles/hour)')
        ax.set_title('Weekly Traffic Patterns by Highway Route')
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self._save_figure('weekly_traffic_patterns')
        plt.close()
        
        print("  → Temporal analysis figures extracted")
    
    def extract_capacity_analysis_figures(self):
        """Create capacity and utilization analysis figures"""
        print("\nExtracting capacity analysis figures...")
        
        # Calculate capacity utilization (assuming 2000 veh/h per lane capacity)
        capacity_per_lane = 2000
        assumed_lanes = 2  # Most highways are 2-lane
        
        self.df['capacity_utilization'] = (self.df['Total_All_Lanes'] / (capacity_per_lane * assumed_lanes)) * 100
        self.df['capacity_utilization'] = np.clip(self.df['capacity_utilization'], 0, 150)  # Cap at 150%
        
        # Figure: Capacity utilization distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self.df['capacity_utilization'].dropna(), bins=50, 
                edgecolor='black', alpha=0.7, color='red')
        ax.axvline(x=80, color='orange', linestyle='--', linewidth=2, label='Critical Threshold (80%)')
        ax.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Full Capacity (100%)')
        ax.set_xlabel('Capacity Utilization (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Highway Capacity Utilization Distribution\nCurrent Traffic vs Theoretical Capacity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self._save_figure('capacity_utilization_distribution')
        plt.close()
        
        # Figure: Monthly capacity trends
        monthly_capacity = self.df.groupby(self.df['datetime'].dt.to_period('M'))['capacity_utilization'].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(monthly_capacity.index.astype(str), monthly_capacity.values, 
               marker='o', linewidth=2.5, color='darkred', alpha=0.8)
        ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Critical Threshold')
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Full Capacity')
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Capacity Utilization (%)')
        ax.set_title('Highway Network Capacity Utilization Trends\nMonthly Averages (2020-2025)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for readability
        ax.tick_params(axis='x', rotation=45)
        
        self._save_figure('capacity_utilization_trends')
        plt.close()
        
        print("  → Capacity analysis figures extracted")
    
    def _save_figure(self, name):
        """Save figure in publication format"""
        self.figure_count += 1
        filename = f"fig_{self.figure_count:02d}_{name}"
        
        # Save as PDF (vector)
        pdf_path = self.figures_dir / f"{filename}.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        
        # Save as PNG (backup)
        png_path = self.figures_dir / f"{filename}.png"
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        
        print(f"    → Saved: {filename}")
    
    def generate_extraction_report(self):
        """Generate summary of extracted figures"""
        
        # Count existing figures
        existing_pdfs = list(self.figures_dir.glob("*.pdf"))
        existing_pngs = list(self.figures_dir.glob("*.png"))
        
        report = {
            'extraction_summary': {
                'total_pdf_figures': len(existing_pdfs),
                'total_png_figures': len(existing_pngs),
                'figures_directory': str(self.figures_dir),
                'extraction_method': 'manual_key_figures'
            },
            'figure_list': sorted([f.stem for f in existing_pdfs])
        }
        
        # Save report
        report_path = Path('/home/niko/workspace/slovenia-trafffic-v2/reports/figure_extraction_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        print(f"\n📊 EXTRACTION SUMMARY")
        print("="*50)
        print(f"PDF figures: {len(existing_pdfs)}")
        print(f"PNG figures: {len(existing_pngs)}")
        print(f"Output directory: {self.figures_dir}")
        print(f"Report saved: {report_path}")

def main():
    """Main extraction function"""
    print("🎨 KEY FIGURE EXTRACTION")
    print("="*50)
    
    extractor = KeyFigureExtractor()
    
    # Load data
    if not extractor.load_data():
        print("❌ Failed to load data")
        return
    
    # Extract figures by category
    extractor.extract_data_exploration_figures()
    extractor.extract_speed_density_figures()  
    extractor.extract_temporal_analysis_figures()
    extractor.extract_capacity_analysis_figures()
    
    # Generate report
    extractor.generate_extraction_report()
    
    print("\n✅ Key figure extraction completed!")

if __name__ == "__main__":
    main()