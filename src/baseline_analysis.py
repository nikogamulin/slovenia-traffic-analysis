"""
Baseline Traffic Pattern Analysis for Slovenian Highway Network
Establishes normal traffic patterns from pre-2024 data, excluding COVID anomalies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BaselineTrafficAnalyzer:
    """
    Analyzes historical traffic data to establish baseline patterns
    before the 2025 traffic collapse
    """
    
    def __init__(self, speed_file, count_file, start_date='2022-01-01', end_date='2023-12-31'):
        """
        Initialize with traffic data files and analysis period
        
        Args:
            speed_file: Path to vehicle speed data
            count_file: Path to vehicle count data
            start_date: Start of baseline period (post-COVID)
            end_date: End of baseline period (pre-roadworks)
        """
        self.speed_file = speed_file
        self.count_file = count_file
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.baseline_patterns = {}
        
    def load_and_filter_data(self):
        """Load traffic data and filter to baseline period"""
        print(f"Loading traffic data from {self.start_date.date()} to {self.end_date.date()}")
        
        # Load speed data
        self.speed_df = pd.read_csv(self.speed_file)
        self.speed_df['date'] = pd.to_datetime(self.speed_df['date'])
        self.speed_df['datetime'] = pd.to_datetime(
            self.speed_df['date'].astype(str) + ' ' + self.speed_df['Time']
        )
        
        # Load count data
        self.count_df = pd.read_csv(self.count_file)
        self.count_df['date'] = pd.to_datetime(self.count_df['date'])
        self.count_df['datetime'] = pd.to_datetime(
            self.count_df['date'].astype(str) + ' ' + self.count_df['Time']
        )
        
        # Filter to baseline period
        self.speed_baseline = self.speed_df[
            (self.speed_df['date'] >= self.start_date) & 
            (self.speed_df['date'] <= self.end_date)
        ].copy()
        
        self.count_baseline = self.count_df[
            (self.count_df['date'] >= self.start_date) & 
            (self.count_df['date'] <= self.end_date)
        ].copy()
        
        print(f"Baseline period: {len(self.speed_baseline)} speed records, "
              f"{len(self.count_baseline)} count records")
        
        # Add temporal features
        self._add_temporal_features()
        
    def _add_temporal_features(self):
        """Add hour, day of week, month features"""
        for df in [self.speed_baseline, self.count_baseline]:
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
            
    def calculate_baseline_patterns(self):
        """Calculate baseline traffic patterns by monitoring point and time"""
        
        # Speed patterns by hour and day of week
        speed_patterns = self.speed_baseline.groupby(
            ['road_name', 'hour', 'day_of_week']
        ).agg({
            'Avg_Speed': ['mean', 'std', 'quantile'],
            'direction_A_avg_speed': ['mean', 'std'],
            'direction_B_avg_speed': ['mean', 'std']
        }).round(2)
        
        # Count patterns by hour and day of week  
        count_patterns = self.count_baseline.groupby(
            ['road_name', 'hour', 'day_of_week']
        ).agg({
            'Total_All_Lanes': ['mean', 'std', 'quantile'],
            'direction_A_count': ['mean', 'std'],
            'direction_B_count': ['mean', 'std'],
            'Trucks_7.5t': ['mean', 'std']
        }).round(0)
        
        # Monthly seasonal patterns
        monthly_speed = self.speed_baseline.groupby(
            ['road_name', 'month']
        )['Avg_Speed'].agg(['mean', 'std']).round(2)
        
        monthly_count = self.count_baseline.groupby(
            ['road_name', 'month']
        )['Total_All_Lanes'].agg(['mean', 'std']).round(0)
        
        self.baseline_patterns = {
            'hourly_speed': speed_patterns,
            'hourly_count': count_patterns,
            'monthly_speed': monthly_speed,
            'monthly_count': monthly_count
        }
        
        print("Baseline patterns calculated for all monitoring points")
        return self.baseline_patterns
        
    def identify_critical_segments(self):
        """Identify segments with highest baseline congestion vulnerability"""
        
        # Calculate congestion metrics
        congestion_metrics = []
        
        for road in self.speed_baseline['road_name'].unique():
            road_data = self.speed_baseline[self.speed_baseline['road_name'] == road]
            
            # Speed variability (coefficient of variation)
            speed_cv = road_data['Avg_Speed'].std() / road_data['Avg_Speed'].mean()
            
            # Frequency of low speeds (< 50 km/h)
            low_speed_freq = (road_data['Avg_Speed'] < 50).mean()
            
            # Peak hour speed drop
            peak_speeds = road_data[road_data['is_peak_hour'] == 1]['Avg_Speed'].mean()
            offpeak_speeds = road_data[road_data['is_peak_hour'] == 0]['Avg_Speed'].mean()
            peak_drop = 1 - (peak_speeds / offpeak_speeds) if offpeak_speeds > 0 else 0
            
            # Traffic volume
            road_counts = self.count_baseline[self.count_baseline['road_name'] == road]
            avg_volume = road_counts['Total_All_Lanes'].mean()
            
            congestion_metrics.append({
                'road_name': road,
                'speed_cv': speed_cv,
                'low_speed_frequency': low_speed_freq,
                'peak_hour_speed_drop': peak_drop,
                'avg_daily_volume': avg_volume,
                'vulnerability_score': speed_cv * low_speed_freq * (1 + peak_drop)
            })
            
        self.congestion_vulnerability = pd.DataFrame(congestion_metrics).sort_values(
            'vulnerability_score', ascending=False
        )
        
        print("\nTop 5 Most Vulnerable Segments (Baseline):")
        print(self.congestion_vulnerability.head()[
            ['road_name', 'vulnerability_score', 'avg_daily_volume']
        ])
        
        return self.congestion_vulnerability
        
    def detect_baseline_anomalies(self):
        """Identify unusual patterns in baseline that might indicate pre-existing issues"""
        
        anomalies = []
        
        for road in self.speed_baseline['road_name'].unique():
            road_speeds = self.speed_baseline[
                self.speed_baseline['road_name'] == road
            ]['Avg_Speed'].values
            
            # Use IQR method for anomaly detection
            Q1 = np.percentile(road_speeds, 25)
            Q3 = np.percentile(road_speeds, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count anomalies
            n_low_anomalies = (road_speeds < lower_bound).sum()
            n_high_anomalies = (road_speeds > upper_bound).sum()
            
            anomalies.append({
                'road_name': road,
                'mean_speed': road_speeds.mean(),
                'std_speed': road_speeds.std(),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_low_speed_anomalies': n_low_anomalies,
                'n_high_speed_anomalies': n_high_anomalies,
                'anomaly_rate': (n_low_anomalies + n_high_anomalies) / len(road_speeds)
            })
            
        self.baseline_anomalies = pd.DataFrame(anomalies).sort_values(
            'anomaly_rate', ascending=False
        )
        
        print("\nSegments with Highest Baseline Anomaly Rates:")
        print(self.baseline_anomalies.head()[
            ['road_name', 'anomaly_rate', 'n_low_speed_anomalies']
        ])
        
        return self.baseline_anomalies
        
    def calculate_spatial_correlations(self):
        """Calculate baseline correlations between monitoring points"""
        
        # Pivot speed data for correlation analysis
        speed_pivot = self.speed_baseline.pivot_table(
            index='datetime',
            columns='road_name',
            values='Avg_Speed',
            aggfunc='mean'
        )
        
        # Calculate correlation matrix
        self.spatial_correlations = speed_pivot.corr()
        
        # Find strongest correlations (excluding self-correlation)
        strong_correlations = []
        for i in range(len(self.spatial_correlations.columns)):
            for j in range(i+1, len(self.spatial_correlations.columns)):
                corr_value = self.spatial_correlations.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'segment_1': self.spatial_correlations.columns[i],
                        'segment_2': self.spatial_correlations.columns[j],
                        'correlation': corr_value
                    })
                    
        self.strong_correlations = pd.DataFrame(strong_correlations).sort_values(
            'correlation', ascending=False
        )
        
        print("\nStrongest Spatial Correlations (Baseline):")
        print(self.strong_correlations.head())
        
        return self.spatial_correlations
        
    def export_baseline_statistics(self, output_dir='./data'):
        """Export baseline statistics for use in other analyses"""
        
        # Summary statistics by road
        summary_stats = []
        
        for road in self.speed_baseline['road_name'].unique():
            road_speeds = self.speed_baseline[
                self.speed_baseline['road_name'] == road
            ]['Avg_Speed']
            
            road_counts = self.count_baseline[
                self.count_baseline['road_name'] == road
            ]['Total_All_Lanes']
            
            summary_stats.append({
                'road_name': road,
                'baseline_mean_speed': road_speeds.mean(),
                'baseline_std_speed': road_speeds.std(),
                'baseline_p10_speed': road_speeds.quantile(0.1),
                'baseline_p50_speed': road_speeds.quantile(0.5),
                'baseline_p90_speed': road_speeds.quantile(0.9),
                'baseline_mean_volume': road_counts.mean() if len(road_counts) > 0 else 0,
                'baseline_std_volume': road_counts.std() if len(road_counts) > 0 else 0,
                'baseline_max_volume': road_counts.max() if len(road_counts) > 0 else 0
            })
            
        self.baseline_summary = pd.DataFrame(summary_stats)
        
        # Save to CSV
        output_file = f"{output_dir}/baseline_statistics_2022_2023.csv"
        self.baseline_summary.to_csv(output_file, index=False)
        print(f"\nBaseline statistics exported to {output_file}")
        
        return self.baseline_summary
        
    def plot_baseline_patterns(self):
        """Generate visualization of baseline patterns"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Average speed by hour of day
        hourly_speed = self.speed_baseline.groupby('hour')['Avg_Speed'].mean()
        axes[0, 0].plot(hourly_speed.index, hourly_speed.values, marker='o')
        axes[0, 0].set_title('Average Speed by Hour (Baseline)')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Speed (km/h)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Traffic volume by hour
        hourly_volume = self.count_baseline.groupby('hour')['Total_All_Lanes'].mean()
        axes[0, 1].bar(hourly_volume.index, hourly_volume.values, color='steelblue')
        axes[0, 1].set_title('Average Traffic Volume by Hour (Baseline)')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Vehicles')
        
        # Plot 3: Speed distribution by day of week
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_speed = self.speed_baseline.groupby('day_of_week')['Avg_Speed'].mean()
        axes[1, 0].bar(range(7), daily_speed.values, tick_label=days, color='green')
        axes[1, 0].set_title('Average Speed by Day of Week (Baseline)')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Speed (km/h)')
        
        # Plot 4: Monthly patterns
        monthly_speed = self.speed_baseline.groupby('month')['Avg_Speed'].mean()
        axes[1, 1].plot(monthly_speed.index, monthly_speed.values, marker='s', color='red')
        axes[1, 1].set_title('Average Speed by Month (Baseline)')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Speed (km/h)')
        axes[1, 1].set_xticks(range(1, 13))
        
        plt.tight_layout()
        plt.savefig('./reports/baseline_patterns.png', dpi=150)
        plt.show()
        
        print("Baseline pattern visualizations saved to ./reports/baseline_patterns.png")


def main():
    """Main execution function"""
    
    # Initialize analyzer
    analyzer = BaselineTrafficAnalyzer(
        speed_file='./data/production_merged_vehicle_speed.csv',
        count_file='./data/production_merged_vehicle_count.csv',
        start_date='2022-01-01',
        end_date='2023-12-31'
    )
    
    # Run baseline analysis
    print("=" * 60)
    print("BASELINE TRAFFIC PATTERN ANALYSIS")
    print("Slovenian Highway Network (2022-2023)")
    print("=" * 60)
    
    # Load and process data
    analyzer.load_and_filter_data()
    
    # Calculate baseline patterns
    baseline_patterns = analyzer.calculate_baseline_patterns()
    
    # Identify vulnerable segments
    vulnerability = analyzer.identify_critical_segments()
    
    # Detect anomalies
    anomalies = analyzer.detect_baseline_anomalies()
    
    # Calculate spatial correlations
    correlations = analyzer.calculate_spatial_correlations()
    
    # Export results
    summary = analyzer.export_baseline_statistics()
    
    # Generate visualizations
    analyzer.plot_baseline_patterns()
    
    print("\n" + "=" * 60)
    print("BASELINE ANALYSIS COMPLETE")
    print("=" * 60)
    
    return analyzer, summary


if __name__ == "__main__":
    analyzer, summary = main()