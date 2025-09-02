"""
Event Study Analysis for Roadwork Impact Assessment
Quantifies direct and ripple effects of major roadwork projects on Slovenian highways
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

class RoadworkEventStudy:
    """
    Conducts interrupted time series analysis for roadwork impacts
    """
    
    def __init__(self, speed_file, count_file, roadworks_file, baseline_stats_file):
        """
        Initialize event study analyzer
        
        Args:
            speed_file: Path to vehicle speed data
            count_file: Path to vehicle count data  
            roadworks_file: Path to roadwork schedule data
            baseline_stats_file: Path to baseline statistics from previous analysis
        """
        self.speed_file = speed_file
        self.count_file = count_file
        self.roadworks_file = roadworks_file
        self.baseline_stats_file = baseline_stats_file
        self.impact_results = {}
        
    def load_data(self):
        """Load all necessary data for event study"""
        
        # Load traffic data
        self.speed_df = pd.read_csv(self.speed_file)
        self.speed_df['date'] = pd.to_datetime(self.speed_df['date'])
        self.speed_df['datetime'] = pd.to_datetime(
            self.speed_df['date'].astype(str) + ' ' + self.speed_df['Time']
        )
        
        self.count_df = pd.read_csv(self.count_file)
        self.count_df['date'] = pd.to_datetime(self.count_df['date'])
        self.count_df['datetime'] = pd.to_datetime(
            self.count_df['date'].astype(str) + ' ' + self.count_df['Time']
        )
        
        # Load roadworks schedule
        self.roadworks = pd.read_csv(self.roadworks_file)
        self.roadworks['start_date'] = pd.to_datetime(self.roadworks['start_date'])
        self.roadworks['end_date'] = pd.to_datetime(self.roadworks['end_date'])
        
        # Load baseline statistics
        self.baseline_stats = pd.read_csv(self.baseline_stats_file)
        
        print(f"Loaded {len(self.roadworks)} roadwork projects")
        print(f"Traffic data spans {self.speed_df['date'].min().date()} to {self.speed_df['date'].max().date()}")
        
    def analyze_roadwork_project(self, project_id, affected_segments, 
                                upstream_segments=None, downstream_segments=None,
                                pre_period_days=90, post_period_days=90):
        """
        Analyze impact of a specific roadwork project
        
        Args:
            project_id: Roadwork project identifier
            affected_segments: List of directly affected road segments
            upstream_segments: List of upstream segments to check for ripple effects
            downstream_segments: List of downstream segments  
            pre_period_days: Days before roadwork to use as control
            post_period_days: Days after roadwork start to analyze
        """
        
        # Get roadwork details
        project = self.roadworks[self.roadworks['roadwork_id'] == project_id].iloc[0]
        start_date = project['start_date']
        end_date = min(project['end_date'], self.speed_df['date'].max())
        
        print(f"\n{'='*60}")
        print(f"Analyzing Project: {project_id}")
        print(f"Location: {project['section_description']}")
        print(f"Management System: {project['management_system']}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"{'='*60}")
        
        results = {
            'project_id': project_id,
            'direct_impacts': {},
            'upstream_ripple': {},
            'downstream_ripple': {}
        }
        
        # Analyze direct impact on affected segments
        for segment in affected_segments:
            impact = self._calculate_segment_impact(
                segment, start_date, pre_period_days, post_period_days
            )
            results['direct_impacts'][segment] = impact
            
        # Analyze upstream ripple effects
        if upstream_segments:
            for segment in upstream_segments:
                impact = self._calculate_segment_impact(
                    segment, start_date, pre_period_days, post_period_days
                )
                results['upstream_ripple'][segment] = impact
                
        # Analyze downstream ripple effects  
        if downstream_segments:
            for segment in downstream_segments:
                impact = self._calculate_segment_impact(
                    segment, start_date, pre_period_days, post_period_days
                )
                results['downstream_ripple'][segment] = impact
                
        self.impact_results[project_id] = results
        
        # Print summary
        self._print_impact_summary(results)
        
        return results
        
    def _calculate_segment_impact(self, segment, intervention_date, 
                                 pre_days=90, post_days=90):
        """
        Calculate impact metrics for a specific segment
        """
        
        # Filter data for segment and time period
        segment_speeds = self.speed_df[
            self.speed_df['road_name'] == segment
        ].copy()
        
        segment_counts = self.count_df[
            self.count_df['road_name'] == segment
        ].copy()
        
        if len(segment_speeds) == 0:
            return {'error': f'No data for segment {segment}'}
            
        # Define pre and post periods
        pre_start = intervention_date - timedelta(days=pre_days)
        post_end = intervention_date + timedelta(days=post_days)
        
        # Get baseline from stored statistics
        baseline_row = self.baseline_stats[
            self.baseline_stats['road_name'] == segment
        ]
        
        if len(baseline_row) > 0:
            baseline_speed = baseline_row.iloc[0]['baseline_mean_speed']
            baseline_volume = baseline_row.iloc[0]['baseline_mean_volume']
        else:
            # Calculate from pre-intervention period if baseline not available
            pre_speeds = segment_speeds[
                (segment_speeds['date'] >= pre_start) & 
                (segment_speeds['date'] < intervention_date)
            ]
            baseline_speed = pre_speeds['Avg_Speed'].mean() if len(pre_speeds) > 0 else 0
            baseline_volume = 0
            
        # Calculate post-intervention metrics
        post_speeds = segment_speeds[
            (segment_speeds['date'] >= intervention_date) & 
            (segment_speeds['date'] <= post_end)
        ]
        
        post_counts = segment_counts[
            (segment_counts['date'] >= intervention_date) & 
            (segment_counts['date'] <= post_end)
        ]
        
        if len(post_speeds) == 0:
            return {'error': 'No post-intervention data'}
            
        # Calculate impact metrics
        post_mean_speed = post_speeds['Avg_Speed'].mean()
        speed_change = post_mean_speed - baseline_speed
        speed_change_pct = (speed_change / baseline_speed * 100) if baseline_speed > 0 else 0
        
        # Volume changes
        post_mean_volume = post_counts['Total_All_Lanes'].mean() if len(post_counts) > 0 else 0
        volume_change = post_mean_volume - baseline_volume if baseline_volume > 0 else 0
        volume_change_pct = (volume_change / baseline_volume * 100) if baseline_volume > 0 else 0
        
        # Statistical significance test
        if len(post_speeds) > 30:
            t_stat, p_value = stats.ttest_1samp(
                post_speeds['Avg_Speed'].values, 
                baseline_speed
            )
        else:
            t_stat, p_value = 0, 1.0
            
        # Peak hour specific impact
        peak_post = post_speeds[post_speeds['datetime'].dt.hour.isin([7, 8, 9, 16, 17, 18])]
        peak_impact = peak_post['Avg_Speed'].mean() - baseline_speed if len(peak_post) > 0 else 0
        
        return {
            'baseline_speed': round(baseline_speed, 1),
            'post_mean_speed': round(post_mean_speed, 1),
            'speed_change_kmh': round(speed_change, 1),
            'speed_change_pct': round(speed_change_pct, 1),
            'baseline_volume': round(baseline_volume, 0),
            'post_mean_volume': round(post_mean_volume, 0),
            'volume_change': round(volume_change, 0),
            'volume_change_pct': round(volume_change_pct, 1),
            'peak_hour_impact_kmh': round(peak_impact, 1),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05
        }
        
    def _print_impact_summary(self, results):
        """Print formatted summary of impact results"""
        
        print("\nDIRECT IMPACTS:")
        print("-" * 50)
        for segment, impact in results['direct_impacts'].items():
            if 'error' not in impact:
                print(f"\n{segment}:")
                print(f"  Speed: {impact['baseline_speed']} → {impact['post_mean_speed']} km/h "
                      f"({impact['speed_change_pct']:+.1f}%)")
                print(f"  Volume: {impact['baseline_volume']:.0f} → {impact['post_mean_volume']:.0f} "
                      f"({impact['volume_change_pct']:+.1f}%)")
                print(f"  Peak Hour Impact: {impact['peak_hour_impact_kmh']:+.1f} km/h")
                print(f"  Statistical Significance: {'Yes' if impact['significant'] else 'No'} "
                      f"(p={impact['p_value']:.4f})")
                      
        if results['upstream_ripple']:
            print("\nUPSTREAM RIPPLE EFFECTS:")
            print("-" * 50)
            for segment, impact in results['upstream_ripple'].items():
                if 'error' not in impact and impact['significant']:
                    print(f"\n{segment}:")
                    print(f"  Speed Change: {impact['speed_change_pct']:+.1f}%")
                    print(f"  Volume Change: {impact['volume_change_pct']:+.1f}%")
                    
        if results['downstream_ripple']:
            print("\nDOWNSTREAM RIPPLE EFFECTS:")
            print("-" * 50)
            for segment, impact in results['downstream_ripple'].items():
                if 'error' not in impact and impact['significant']:
                    print(f"\n{segment}:")
                    print(f"  Speed Change: {impact['speed_change_pct']:+.1f}%")
                    print(f"  Volume Change: {impact['volume_change_pct']:+.1f}%")
                    
    def compare_management_systems(self):
        """Compare effectiveness of different traffic management systems"""
        
        management_impacts = {}
        
        for project_id, results in self.impact_results.items():
            project = self.roadworks[self.roadworks['roadwork_id'] == project_id].iloc[0]
            management_system = project['management_system']
            
            if management_system not in management_impacts:
                management_impacts[management_system] = []
                
            # Aggregate direct impacts
            for segment, impact in results['direct_impacts'].items():
                if 'error' not in impact:
                    management_impacts[management_system].append({
                        'project': project_id,
                        'speed_change_pct': impact['speed_change_pct'],
                        'volume_change_pct': impact['volume_change_pct']
                    })
                    
        # Calculate average impacts by management system
        print("\n" + "=" * 60)
        print("TRAFFIC MANAGEMENT SYSTEM COMPARISON")
        print("=" * 60)
        
        for system, impacts in management_impacts.items():
            if impacts:
                avg_speed_impact = np.mean([i['speed_change_pct'] for i in impacts])
                avg_volume_impact = np.mean([i['volume_change_pct'] for i in impacts])
                
                print(f"\n{system}:")
                print(f"  Average Speed Impact: {avg_speed_impact:+.1f}%")
                print(f"  Average Volume Impact: {avg_volume_impact:+.1f}%")
                print(f"  Number of Projects: {len(set([i['project'] for i in impacts]))}")
                
    def calculate_ripple_decay(self):
        """Calculate how impact decays with distance from roadwork"""
        
        ripple_data = []
        
        for project_id, results in self.impact_results.items():
            # Direct impact (distance = 0)
            for segment, impact in results['direct_impacts'].items():
                if 'error' not in impact:
                    ripple_data.append({
                        'project': project_id,
                        'distance': 0,
                        'impact_type': 'direct',
                        'speed_impact': abs(impact['speed_change_pct'])
                    })
                    
            # Upstream impacts (distance = 1)
            for segment, impact in results['upstream_ripple'].items():
                if 'error' not in impact:
                    ripple_data.append({
                        'project': project_id,
                        'distance': 1,
                        'impact_type': 'upstream',
                        'speed_impact': abs(impact['speed_change_pct'])
                    })
                    
            # Downstream impacts (distance = 1)  
            for segment, impact in results['downstream_ripple'].items():
                if 'error' not in impact:
                    ripple_data.append({
                        'project': project_id,
                        'distance': 1,
                        'impact_type': 'downstream',
                        'speed_impact': abs(impact['speed_change_pct'])
                    })
                    
        if ripple_data:
            ripple_df = pd.DataFrame(ripple_data)
            
            # Calculate average decay
            decay_summary = ripple_df.groupby('distance')['speed_impact'].agg(['mean', 'std'])
            
            print("\n" + "=" * 60)
            print("RIPPLE EFFECT DECAY ANALYSIS")
            print("=" * 60)
            print("\nAverage Speed Impact by Distance from Roadwork:")
            print(decay_summary)
            
            # Calculate decay rate
            if len(decay_summary) > 1:
                decay_rate = (decay_summary.iloc[0]['mean'] - decay_summary.iloc[1]['mean']) / decay_summary.iloc[0]['mean']
                print(f"\nRipple Effect Decay Rate: {decay_rate:.1%} per segment")
                
        return ripple_data
        
    def export_results(self, output_file='./data/event_study_results.csv'):
        """Export detailed event study results"""
        
        results_list = []
        
        for project_id, results in self.impact_results.items():
            project = self.roadworks[self.roadworks['roadwork_id'] == project_id].iloc[0]
            
            # Export direct impacts
            for segment, impact in results['direct_impacts'].items():
                if 'error' not in impact:
                    results_list.append({
                        'project_id': project_id,
                        'section': project['section_description'],
                        'management_system': project['management_system'],
                        'impact_type': 'direct',
                        'segment': segment,
                        'speed_change_pct': impact['speed_change_pct'],
                        'volume_change_pct': impact['volume_change_pct'],
                        'peak_hour_impact': impact['peak_hour_impact_kmh'],
                        'significant': impact['significant']
                    })
                    
            # Export ripple effects
            for ripple_type, ripple_dict in [('upstream', results['upstream_ripple']), 
                                             ('downstream', results['downstream_ripple'])]:
                for segment, impact in ripple_dict.items():
                    if 'error' not in impact:
                        results_list.append({
                            'project_id': project_id,
                            'section': project['section_description'],
                            'management_system': project['management_system'],
                            'impact_type': ripple_type,
                            'segment': segment,
                            'speed_change_pct': impact['speed_change_pct'],
                            'volume_change_pct': impact['volume_change_pct'],
                            'peak_hour_impact': impact['peak_hour_impact_kmh'],
                            'significant': impact['significant']
                        })
                        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_file, index=False)
        print(f"\nEvent study results exported to {output_file}")
        
        return results_df


def main():
    """Main execution for event study analysis"""
    
    # Initialize analyzer
    analyzer = RoadworkEventStudy(
        speed_file='./data/production_merged_vehicle_speed.csv',
        count_file='./data/production_merged_vehicle_count.csv',
        roadworks_file='./data/external/roadworks/roadworks_actual_2024_2026.csv',
        baseline_stats_file='./data/baseline_statistics_2022_2023.csv'
    )
    
    # Load data
    analyzer.load_data()
    
    print("\n" + "=" * 60)
    print("EVENT STUDY ANALYSIS: ROADWORK IMPACTS")
    print("=" * 60)
    
    # Analyze major roadwork projects
    
    # Project 1: A1 Slovenske Konjice - Dramlje (1+1+1 bidirectional)
    analyzer.analyze_roadwork_project(
        project_id='RW_DARS_001',
        affected_segments=['Celje-Maribor'],
        upstream_segments=['Ljubljana-Celje'],
        downstream_segments=['Maribor HC', 'Maribor-Ptuj']
    )
    
    # Project 2: A1 Kozina - Črni Kal (lane closures)
    analyzer.analyze_roadwork_project(
        project_id='RW_DARS_003',
        affected_segments=['Postojna-Koper'],
        upstream_segments=['Koper-Ljubljana'],
        downstream_segments=['Koper Port', 'Bertoki HC']
    )
    
    # Project 3: Podravska Region repairs (mixed closures)
    analyzer.analyze_roadwork_project(
        project_id='RW_DRSI_001',
        affected_segments=['Maribor-Ptuj'],
        upstream_segments=['Celje-Maribor'],
        downstream_segments=['Murska Sobota HC']
    )
    
    # Compare management systems
    analyzer.compare_management_systems()
    
    # Analyze ripple decay
    ripple_data = analyzer.calculate_ripple_decay()
    
    # Export results
    results_df = analyzer.export_results()
    
    print("\n" + "=" * 60)
    print("EVENT STUDY ANALYSIS COMPLETE")
    print("=" * 60)
    
    return analyzer, results_df


if __name__ == "__main__":
    analyzer, results = main()