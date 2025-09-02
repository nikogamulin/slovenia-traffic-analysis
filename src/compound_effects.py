"""
Compound Effects Analysis for Slovenian Traffic Collapse
Analyzes interaction effects between roadworks, holidays, incidents, and weather
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class CompoundEffectsAnalyzer:
    """
    Analyzes compounding factors that amplified the traffic collapse
    """
    
    def __init__(self, speed_file, count_file, roadworks_file, 
                 incidents_file, holidays_file, weather_file):
        """
        Initialize compound effects analyzer
        
        Args:
            speed_file: Path to vehicle speed data
            count_file: Path to vehicle count data
            roadworks_file: Path to roadwork schedule
            incidents_file: Path to incident data
            holidays_file: Path to holiday calendar
            weather_file: Path to weather data
        """
        self.speed_file = speed_file
        self.count_file = count_file
        self.roadworks_file = roadworks_file
        self.incidents_file = incidents_file
        self.holidays_file = holidays_file
        self.weather_file = weather_file
        self.compound_results = {}
        
    def load_and_merge_data(self):
        """Load and merge all data sources"""
        
        print("Loading and merging multi-source data...")
        
        # Load traffic data
        self.speed_df = pd.read_csv(self.speed_file)
        self.speed_df['date'] = pd.to_datetime(self.speed_df['date'])
        
        # Load roadworks
        self.roadworks = pd.read_csv(self.roadworks_file)
        self.roadworks['start_date'] = pd.to_datetime(self.roadworks['start_date'])
        self.roadworks['end_date'] = pd.to_datetime(self.roadworks['end_date'])
        
        # Load incidents
        self.incidents = pd.read_csv(self.incidents_file)
        self.incidents['date'] = pd.to_datetime(self.incidents['date'])
        
        # Load holidays
        self.holidays = pd.read_csv(self.holidays_file)
        self.holidays['date'] = pd.to_datetime(self.holidays['date'])
        
        # Load weather
        self.weather = pd.read_csv(self.weather_file)
        self.weather['datetime'] = pd.to_datetime(self.weather['datetime'])
        self.weather['date'] = self.weather['datetime'].dt.date
        self.weather['date'] = pd.to_datetime(self.weather['date'])
        
        # Create daily aggregates
        self._create_daily_features()
        
        print(f"Data loaded: {len(self.daily_data)} days of integrated data")
        
    def _create_daily_features(self):
        """Create daily feature matrix combining all data sources"""
        
        # Daily traffic aggregates
        daily_speed = self.speed_df.groupby(['date', 'road_name']).agg({
            'Avg_Speed': ['mean', 'std', 'min'],
            'direction_A_avg_speed': 'mean',
            'direction_B_avg_speed': 'mean'
        }).reset_index()
        daily_speed.columns = ['date', 'road_name', 'avg_speed', 'speed_std', 
                              'min_speed', 'dirA_speed', 'dirB_speed']
        
        # Count active roadworks per day
        dates = pd.date_range(self.speed_df['date'].min(), 
                             self.speed_df['date'].max(), freq='D')
        
        roadwork_counts = []
        for date in dates:
            active = self.roadworks[
                (self.roadworks['start_date'] <= date) & 
                (self.roadworks['end_date'] >= date)
            ]
            
            roadwork_counts.append({
                'date': date,
                'n_major_roadworks': len(active[active['impact_level'] == 'Severe']),
                'n_moderate_roadworks': len(active[active['impact_level'] == 'Moderate']),
                'n_total_roadworks': len(active),
                'has_bidirectional': any(active['management_system'].str.contains('bidirectional', na=False))
            })
            
        roadwork_df = pd.DataFrame(roadwork_counts)
        
        # Daily incident counts and severity
        daily_incidents = self.incidents.groupby('date').agg({
            'incident_id': 'count',
            'clearance_minutes': ['mean', 'max'],
            'severity': lambda x: (x == 'Major').sum()
        }).reset_index()
        daily_incidents.columns = ['date', 'n_incidents', 'avg_clearance', 
                                  'max_clearance', 'n_major_incidents']
        
        # Holiday indicators
        holiday_indicators = self.holidays.groupby('date').agg({
            'country': lambda x: list(x.unique()),
            'type': lambda x: 'school' in ' '.join(x).lower()
        }).reset_index()
        holiday_indicators.columns = ['date', 'holiday_countries', 'is_school_holiday']
        holiday_indicators['is_international_holiday'] = holiday_indicators['holiday_countries'].apply(
            lambda x: len(x) > 1
        )
        
        # Daily weather aggregates
        daily_weather = self.weather.groupby('date').agg({
            'precipitation_mm': ['sum', 'max'],
            'temperature_c': ['mean', 'min'],
            'wind_speed_kmh': 'max',
            'visibility_m': 'min'
        }).reset_index()
        daily_weather.columns = ['date', 'total_precip', 'max_precip', 
                                'avg_temp', 'min_temp', 'max_wind', 'min_visibility']
        daily_weather['is_adverse_weather'] = (
            (daily_weather['total_precip'] > 10) | 
            (daily_weather['min_visibility'] < 1000) |
            (daily_weather['max_wind'] > 50)
        ).astype(int)
        
        # Merge all features
        self.daily_data = daily_speed
        for df in [roadwork_df, daily_incidents, holiday_indicators, daily_weather]:
            self.daily_data = self.daily_data.merge(df, on='date', how='left')
            
        # Fill missing values
        self.daily_data = self.daily_data.fillna(0)
        
        # Add temporal features
        self.daily_data['day_of_week'] = self.daily_data['date'].dt.dayofweek
        self.daily_data['month'] = self.daily_data['date'].dt.month
        self.daily_data['is_weekend'] = self.daily_data['day_of_week'].isin([5, 6]).astype(int)
        
    def analyze_simultaneous_roadworks(self):
        """Analyze periods with multiple simultaneous roadworks"""
        
        print("\n" + "=" * 60)
        print("SIMULTANEOUS ROADWORKS ANALYSIS")
        print("=" * 60)
        
        # Focus on 2025 when collapse occurred
        data_2025 = self.daily_data[self.daily_data['date'].dt.year == 2025].copy()
        
        # Categorize by number of active roadworks
        data_2025['roadwork_category'] = pd.cut(
            data_2025['n_total_roadworks'],
            bins=[-1, 0, 2, 4, 100],
            labels=['None', 'Low (1-2)', 'Medium (3-4)', 'High (5+)']
        )
        
        # Calculate average speed by roadwork intensity
        impact_by_intensity = data_2025.groupby('roadwork_category').agg({
            'avg_speed': ['mean', 'std'],
            'n_incidents': 'mean',
            'date': 'count'
        }).round(2)
        
        print("\nImpact by Number of Simultaneous Roadworks:")
        print(impact_by_intensity)
        
        # Test if impact is additive or multiplicative
        baseline_speed = data_2025[data_2025['n_total_roadworks'] == 0]['avg_speed'].mean()
        
        impacts = []
        for n in range(1, 6):
            subset = data_2025[data_2025['n_total_roadworks'] == n]
            if len(subset) > 0:
                avg_speed = subset['avg_speed'].mean()
                impact_pct = (baseline_speed - avg_speed) / baseline_speed * 100
                impacts.append({
                    'n_roadworks': n,
                    'avg_speed': avg_speed,
                    'impact_pct': impact_pct,
                    'n_days': len(subset)
                })
                
        impacts_df = pd.DataFrame(impacts)
        
        if len(impacts_df) > 2:
            # Fit linear and exponential models to test relationship
            from scipy.optimize import curve_fit
            
            def linear_model(x, a, b):
                return a * x + b
                
            def exponential_model(x, a, b, c):
                return a * np.exp(b * x) + c
                
            x = impacts_df['n_roadworks'].values
            y = impacts_df['impact_pct'].values
            
            # Fit models
            linear_params, _ = curve_fit(linear_model, x, y)
            exp_params, _ = curve_fit(exponential_model, x, y, maxfev=5000)
            
            # Calculate R-squared
            linear_pred = linear_model(x, *linear_params)
            exp_pred = exponential_model(x, *exp_params)
            
            linear_r2 = 1 - np.sum((y - linear_pred)**2) / np.sum((y - y.mean())**2)
            exp_r2 = 1 - np.sum((y - exp_pred)**2) / np.sum((y - y.mean())**2)
            
            print(f"\nImpact Scaling Analysis:")
            print(f"Linear Model R²: {linear_r2:.3f}")
            print(f"Exponential Model R²: {exp_r2:.3f}")
            
            if exp_r2 > linear_r2 + 0.1:
                print("→ Impact appears MULTIPLICATIVE (exponential scaling)")
            else:
                print("→ Impact appears ADDITIVE (linear scaling)")
                
        self.compound_results['simultaneous_roadworks'] = impacts_df
        
        return impacts_df
        
    def analyze_holiday_roadwork_interaction(self):
        """Analyze interaction between holidays and roadworks"""
        
        print("\n" + "=" * 60)
        print("HOLIDAY-ROADWORK INTERACTION ANALYSIS")
        print("=" * 60)
        
        # Create interaction categories
        self.daily_data['scenario'] = 'Normal'
        self.daily_data.loc[
            (self.daily_data['n_total_roadworks'] > 0) & 
            (self.daily_data['is_international_holiday'] == 0),
            'scenario'
        ] = 'Roadwork Only'
        self.daily_data.loc[
            (self.daily_data['n_total_roadworks'] == 0) & 
            (self.daily_data['is_international_holiday'] == 1),
            'scenario'
        ] = 'Holiday Only'
        self.daily_data.loc[
            (self.daily_data['n_total_roadworks'] > 0) & 
            (self.daily_data['is_international_holiday'] == 1),
            'scenario'
        ] = 'Roadwork + Holiday'
        
        # Calculate impacts
        scenario_impact = self.daily_data.groupby('scenario').agg({
            'avg_speed': ['mean', 'std'],
            'n_incidents': 'mean',
            'date': 'count'
        }).round(2)
        
        print("\nTraffic Impact by Scenario:")
        print(scenario_impact)
        
        # Calculate interaction effect
        normal_speed = self.daily_data[
            self.daily_data['scenario'] == 'Normal'
        ]['avg_speed'].mean()
        
        roadwork_only = self.daily_data[
            self.daily_data['scenario'] == 'Roadwork Only'
        ]['avg_speed'].mean()
        
        holiday_only = self.daily_data[
            self.daily_data['scenario'] == 'Holiday Only'
        ]['avg_speed'].mean()
        
        combined = self.daily_data[
            self.daily_data['scenario'] == 'Roadwork + Holiday'
        ]['avg_speed'].mean()
        
        # Expected additive effect
        expected_additive = normal_speed - (normal_speed - roadwork_only) - (normal_speed - holiday_only)
        actual_combined = combined
        
        interaction_effect = expected_additive - actual_combined
        
        print(f"\nInteraction Effect Analysis:")
        print(f"Normal Speed: {normal_speed:.1f} km/h")
        print(f"Roadwork Only: {roadwork_only:.1f} km/h (Δ = {roadwork_only - normal_speed:.1f})")
        print(f"Holiday Only: {holiday_only:.1f} km/h (Δ = {holiday_only - normal_speed:.1f})")
        print(f"Combined Actual: {actual_combined:.1f} km/h")
        print(f"Expected (if additive): {expected_additive:.1f} km/h")
        print(f"Interaction Effect: {interaction_effect:.1f} km/h")
        
        if interaction_effect > 5:
            print("→ SYNERGISTIC interaction (worse than additive)")
        elif interaction_effect < -5:
            print("→ ANTAGONISTIC interaction (better than additive)")
        else:
            print("→ ADDITIVE interaction (no significant interaction)")
            
        self.compound_results['holiday_interaction'] = {
            'normal_speed': normal_speed,
            'roadwork_impact': roadwork_only - normal_speed,
            'holiday_impact': holiday_only - normal_speed,
            'combined_impact': actual_combined - normal_speed,
            'interaction_effect': interaction_effect
        }
        
        return interaction_effect
        
    def analyze_incident_clustering(self):
        """Analyze if incidents cluster near roadwork zones"""
        
        print("\n" + "=" * 60)
        print("INCIDENT CLUSTERING ANALYSIS")
        print("=" * 60)
        
        # Get roadwork periods and locations
        roadwork_periods = []
        for _, rw in self.roadworks.iterrows():
            roadwork_periods.append({
                'road_code': rw['road_code'],
                'start': rw['start_date'],
                'end': rw['end_date'],
                'project': rw['roadwork_id']
            })
            
        # Classify incidents as near/far from roadworks
        incident_classification = []
        
        for _, incident in self.incidents.iterrows():
            near_roadwork = False
            for rw in roadwork_periods:
                if (incident['date'] >= rw['start'] and 
                    incident['date'] <= rw['end'] and
                    incident['road_code'] == rw['road_code']):
                    near_roadwork = True
                    break
                    
            incident_classification.append({
                'date': incident['date'],
                'severity': incident['severity'],
                'clearance': incident['clearance_minutes'],
                'near_roadwork': near_roadwork
            })
            
        incidents_class = pd.DataFrame(incident_classification)
        
        # Compare incident rates
        near_rw = incidents_class[incidents_class['near_roadwork']]
        far_rw = incidents_class[~incidents_class['near_roadwork']]
        
        print(f"\nIncidents Near Roadworks: {len(near_rw)}")
        print(f"Incidents Away from Roadworks: {len(far_rw)}")
        
        # Statistical test for difference in severity
        if len(near_rw) > 0 and len(far_rw) > 0:
            # Compare clearance times
            t_stat, p_value = stats.ttest_ind(
                near_rw['clearance'].values,
                far_rw['clearance'].values
            )
            
            print(f"\nAverage Clearance Time:")
            print(f"  Near Roadworks: {near_rw['clearance'].mean():.1f} minutes")
            print(f"  Away from Roadworks: {far_rw['clearance'].mean():.1f} minutes")
            print(f"  Statistical Test: p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                print("→ SIGNIFICANT difference in incident characteristics near roadworks")
            else:
                print("→ No significant difference in incident characteristics")
                
            # Compare severity distribution
            near_severe = (near_rw['severity'] == 'Major').mean()
            far_severe = (far_rw['severity'] == 'Major').mean()
            
            print(f"\nProportion of Major Incidents:")
            print(f"  Near Roadworks: {near_severe:.1%}")
            print(f"  Away from Roadworks: {far_severe:.1%}")
            
        self.compound_results['incident_clustering'] = {
            'n_near_roadwork': len(near_rw),
            'n_far_roadwork': len(far_rw),
            'avg_clearance_near': near_rw['clearance'].mean() if len(near_rw) > 0 else 0,
            'avg_clearance_far': far_rw['clearance'].mean() if len(far_rw) > 0 else 0,
            'p_value': p_value if len(near_rw) > 0 and len(far_rw) > 0 else 1.0
        }
        
        return incidents_class
        
    def identify_perfect_storm_days(self):
        """Identify days with multiple compounding factors"""
        
        print("\n" + "=" * 60)
        print("PERFECT STORM ANALYSIS")
        print("=" * 60)
        
        # Define perfect storm conditions
        self.daily_data['storm_score'] = (
            (self.daily_data['n_major_roadworks'] > 0).astype(int) * 2 +
            (self.daily_data['n_total_roadworks'] > 2).astype(int) +
            (self.daily_data['is_international_holiday']).astype(int) +
            (self.daily_data['is_adverse_weather']).astype(int) +
            (self.daily_data['n_major_incidents'] > 0).astype(int) +
            (self.daily_data['is_weekend']).astype(int) * 0.5
        )
        
        # Find worst days
        worst_days = self.daily_data.nlargest(10, 'storm_score')[
            ['date', 'road_name', 'avg_speed', 'storm_score', 
             'n_total_roadworks', 'n_incidents', 'is_international_holiday']
        ]
        
        print("\nTop 10 'Perfect Storm' Days:")
        print(worst_days)
        
        # Analyze speed distribution by storm score
        storm_impact = self.daily_data.groupby(
            pd.cut(self.daily_data['storm_score'], bins=5)
        )['avg_speed'].agg(['mean', 'std', 'count'])
        
        print("\nSpeed Impact by Storm Score:")
        print(storm_impact)
        
        # Calculate correlation between storm score and speed
        correlation = self.daily_data['storm_score'].corr(self.daily_data['avg_speed'])
        print(f"\nCorrelation between Storm Score and Speed: {correlation:.3f}")
        
        self.compound_results['perfect_storms'] = worst_days
        
        return worst_days
        
    def export_compound_analysis(self, output_dir='./data'):
        """Export compound effects analysis results"""
        
        # Export daily integrated data
        daily_file = f"{output_dir}/daily_compound_factors.csv"
        self.daily_data.to_csv(daily_file, index=False)
        print(f"\nDaily compound factors exported to {daily_file}")
        
        # Export summary results
        summary = pd.DataFrame([
            {
                'metric': 'Max Simultaneous Roadworks',
                'value': self.daily_data['n_total_roadworks'].max()
            },
            {
                'metric': 'Days with 3+ Roadworks',
                'value': (self.daily_data['n_total_roadworks'] >= 3).sum()
            },
            {
                'metric': 'Holiday-Roadwork Interaction Effect',
                'value': self.compound_results.get('holiday_interaction', {}).get('interaction_effect', 0)
            },
            {
                'metric': 'Incident Increase Near Roadworks',
                'value': (
                    self.compound_results.get('incident_clustering', {}).get('avg_clearance_near', 0) -
                    self.compound_results.get('incident_clustering', {}).get('avg_clearance_far', 0)
                )
            },
            {
                'metric': 'Max Storm Score',
                'value': self.daily_data['storm_score'].max()
            }
        ])
        
        summary_file = f"{output_dir}/compound_effects_summary.csv"
        summary.to_csv(summary_file, index=False)
        print(f"Compound effects summary exported to {summary_file}")
        
        return summary


def main():
    """Main execution for compound effects analysis"""
    
    # Initialize analyzer
    analyzer = CompoundEffectsAnalyzer(
        speed_file='./data/production_merged_vehicle_speed.csv',
        count_file='./data/production_merged_vehicle_count.csv',
        roadworks_file='./data/external/roadworks/roadworks_actual_2024_2026.csv',
        incidents_file='./data/external/incidents/accident_data_2020_2025.csv',
        holidays_file='./data/external/holidays/holidays_combined_2020_2025.csv',
        weather_file='./data/external/weather/arso_weather_2020_2025.csv'
    )
    
    print("\n" + "=" * 60)
    print("COMPOUND EFFECTS ANALYSIS")
    print("Slovenian Traffic Collapse 2025")
    print("=" * 60)
    
    # Load and merge data
    analyzer.load_and_merge_data()
    
    # Analyze simultaneous roadworks
    roadwork_impacts = analyzer.analyze_simultaneous_roadworks()
    
    # Analyze holiday interactions
    holiday_interaction = analyzer.analyze_holiday_roadwork_interaction()
    
    # Analyze incident clustering
    incident_analysis = analyzer.analyze_incident_clustering()
    
    # Identify perfect storm days
    worst_days = analyzer.identify_perfect_storm_days()
    
    # Export results
    summary = analyzer.export_compound_analysis()
    
    print("\n" + "=" * 60)
    print("COMPOUND EFFECTS ANALYSIS COMPLETE")
    print("=" * 60)
    
    return analyzer, summary


if __name__ == "__main__":
    analyzer, summary = main()