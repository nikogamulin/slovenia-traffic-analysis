#!/usr/bin/env python3
"""
Economic Impact Assessment Figure Extraction
Creates publication-ready economic analysis figures

Task 6.3: Economic Analysis Figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Publication styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class EconomicFigureExtractor:
    """Extract economic impact assessment figures"""
    
    def __init__(self):
        self.figures_dir = Path('/home/niko/workspace/slovenia-trafffic-v2/reports/article/figures')
        self.figure_count = 33  # Continue from previous extraction
        
        # Economic data from analysis
        self.economic_data = {
            'direct_costs': {
                'congestion_delays': 287,
                'fuel_consumption': 43,
                'traffic_accidents': 142,
                'co2_emissions': 52,
                'air_quality': 24,
                'productivity_loss': -43  # Negative = avoided cost
            },
            'investment': {
                'capital_cost': 550,
                'annual_om': 28,
                'npv_10yr': 4321
            },
            'metrics': {
                'bcr': 4.8,
                'irr': 0.72,
                'payback_years': 1.4,
                'breakeven_year': 2027
            }
        }
    
    def create_cost_breakdown_waterfall(self):
        """Create waterfall chart of economic impacts"""
        print("Creating economic cost breakdown waterfall...")
        
        costs = self.economic_data['direct_costs']
        
        # Waterfall data
        categories = ['Congestion\nDelays', 'Fuel\nConsumption', 'Traffic\nAccidents', 
                     'CO₂\nEmissions', 'Air\nQuality', 'Productivity\nSavings', 'Total']
        values = [costs['congestion_delays'], costs['fuel_consumption'], costs['traffic_accidents'],
                 costs['co2_emissions'], costs['air_quality'], costs['productivity_loss'], 505]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors for different categories
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#A29BFE', '#2D3436']
        
        # Create waterfall bars
        cumulative = 0
        bar_positions = []
        bar_heights = []
        bar_bottoms = []
        bar_colors = []
        
        for i, (cat, val) in enumerate(zip(categories[:-1], values[:-1])):
            if i == 0:
                # First bar starts from 0
                bar_bottoms.append(0)
                bar_heights.append(val)
                bar_positions.append(i)
                cumulative = val
            else:
                if val >= 0:
                    bar_bottoms.append(cumulative)
                    bar_heights.append(val)
                else:
                    bar_bottoms.append(cumulative + val)
                    bar_heights.append(abs(val))
                cumulative += val
                bar_positions.append(i)
            
            bar_colors.append(colors[i])
        
        # Add total bar
        bar_positions.append(len(categories) - 1)
        bar_bottoms.append(0)
        bar_heights.append(cumulative)
        bar_colors.append(colors[-1])
        
        # Create bars
        bars = ax.bar(bar_positions, bar_heights, bottom=bar_bottoms, 
                     color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (pos, height, bottom) in enumerate(zip(bar_positions, bar_heights, bar_bottoms)):
            if i < len(values) - 1:
                label_y = bottom + height/2
                ax.text(pos, label_y, f'€{abs(values[i])}M', 
                       ha='center', va='center', fontweight='bold', fontsize=11)
            else:
                ax.text(pos, height/2, f'€{cumulative:.0f}M\nTotal', 
                       ha='center', va='center', fontweight='bold', fontsize=12)
        
        # Add connecting lines
        for i in range(len(bar_positions) - 2):
            start_x = bar_positions[i] + 0.4
            end_x = bar_positions[i + 1] - 0.4
            y = bar_bottoms[i] + bar_heights[i] if values[i] >= 0 else bar_bottoms[i]
            
            ax.plot([start_x, end_x], [y, y], 'k--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Impact Categories')
        ax.set_ylabel('Annual Economic Impact (€ Millions)')
        ax.set_title('Economic Impact Breakdown - Direct Highway Costs\nAnnual Costs vs Highway Expansion Benefits', 
                    fontweight='bold')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(categories, rotation=0, ha='center')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add annotation
        ax.text(0.02, 0.98, 'Note: Negative values represent cost savings from intervention', 
                transform=ax.transAxes, fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self._save_figure('economic_cost_waterfall')
        plt.close()
        
        print("  → Economic waterfall chart created")
    
    def create_benefit_cost_analysis(self):
        """Create benefit-cost analysis visualization"""
        print("Creating benefit-cost analysis...")
        
        # Investment timeline data
        years = np.arange(2025, 2035)
        annual_benefits = 505  # Million euros
        capital_cost = self.economic_data['investment']['capital_cost']
        annual_om = self.economic_data['investment']['annual_om']
        
        # Calculate NPV timeline
        discount_rate = 0.03
        cumulative_benefits = []
        cumulative_costs = []
        net_benefits = []
        
        for i, year in enumerate(years):
            if i == 0:
                # Year 0: Capital investment
                cum_cost = capital_cost
                cum_benefit = 0
            else:
                # Annual benefits and O&M costs
                annual_benefit_pv = annual_benefits / ((1 + discount_rate) ** i)
                annual_cost_pv = annual_om / ((1 + discount_rate) ** i)
                
                cum_benefit = cumulative_benefits[-1] + annual_benefit_pv if cumulative_benefits else annual_benefit_pv
                cum_cost = cumulative_costs[-1] + annual_cost_pv if cumulative_costs else capital_cost + annual_cost_pv
            
            cumulative_benefits.append(cum_benefit)
            cumulative_costs.append(cum_cost)
            net_benefits.append(cum_benefit - cum_cost)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cumulative benefits vs costs
        axes[0,0].plot(years, cumulative_benefits, marker='o', linewidth=3, 
                      color='green', label='Cumulative Benefits')
        axes[0,0].plot(years, cumulative_costs, marker='s', linewidth=3, 
                      color='red', label='Cumulative Costs')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Present Value (€ Millions)')
        axes[0,0].set_title('Cumulative Benefits vs Costs (NPV)', fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Find breakeven point
        breakeven_idx = next((i for i, nb in enumerate(net_benefits) if nb > 0), len(net_benefits))
        if breakeven_idx < len(years):
            axes[0,0].axvline(years[breakeven_idx], color='orange', linestyle='--', 
                             linewidth=2, label=f'Breakeven: {years[breakeven_idx]}')
            axes[0,0].legend()
        
        # 2. Net benefits over time
        axes[0,1].bar(years, net_benefits, color=['red' if x < 0 else 'green' for x in net_benefits],
                     alpha=0.7, edgecolor='black')
        axes[0,1].axhline(0, color='black', linewidth=1)
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Net Benefits (€ Millions)')
        axes[0,1].set_title('Annual Net Benefits Timeline', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. ROI timeline
        roi_values = [(b - c) / c * 100 if c > 0 else 0 for b, c in zip(cumulative_benefits, cumulative_costs)]
        axes[1,0].plot(years, roi_values, marker='d', linewidth=3, color='purple')
        axes[1,0].axhline(0, color='black', linestyle='--', alpha=0.7)
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Return on Investment (%)')
        axes[1,0].set_title('ROI Timeline', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Key metrics dashboard
        metrics = self.economic_data['metrics']
        metric_names = ['Benefit-Cost\nRatio', 'Internal Rate\nof Return (%)', 
                       'Payback\nPeriod (years)', 'NPV 10-year\n(€M)']
        metric_values = [metrics['bcr'], metrics['irr'] * 100, 
                        metrics['payback_years'], self.economic_data['investment']['npv_10yr']]
        metric_colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
        
        bars = axes[1,1].bar(range(len(metric_names)), metric_values, 
                            color=metric_colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height/2,
                          f'{value:.1f}', ha='center', va='center', 
                          fontweight='bold', fontsize=11, color='white')
        
        axes[1,1].set_xticks(range(len(metric_names)))
        axes[1,1].set_xticklabels(metric_names)
        axes[1,1].set_ylabel('Value')
        axes[1,1].set_title('Key Financial Metrics', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_figure('benefit_cost_analysis')
        plt.close()
        
        print("  → Benefit-cost analysis created")
    
    def create_impact_comparison(self):
        """Create comparison of direct vs total economic impact"""
        print("Creating impact comparison visualization...")
        
        # Data: Direct highway costs vs total economic impact
        categories = ['Direct Highway\nCosts', 'Network-wide\nEffects', 'Indirect Economic\nImpact']
        values = [505, 1265, 600]  # Million euros
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Stacked bar chart
        cumulative = np.cumsum([0] + values)
        
        bars = []
        for i, (cat, val, color) in enumerate(zip(categories, values, colors)):
            bar = axes[0].bar(0, val, bottom=cumulative[i], color=color, 
                             alpha=0.8, edgecolor='black', linewidth=1, 
                             width=0.6, label=cat)
            bars.append(bar)
            
            # Add value labels
            axes[0].text(0, cumulative[i] + val/2, f'€{val}M\n({val/sum(values)*100:.1f}%)', 
                        ha='center', va='center', fontweight='bold', fontsize=12)
        
        axes[0].set_ylabel('Economic Impact (€ Millions)')
        axes[0].set_title('Total Annual Economic Impact Breakdown\n€2.37 Billion Total', 
                         fontweight='bold')
        axes[0].set_xlim(-0.5, 0.5)
        axes[0].set_xticks([])
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add total at top
        axes[0].text(0, sum(values) + 50, f'Total: €{sum(values):,}M', 
                    ha='center', va='bottom', fontweight='bold', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 2. Comparison pie chart
        wedges, texts, autotexts = axes[1].pie(values, labels=categories, colors=colors,
                                              autopct='%1.1f%%', startangle=90,
                                              explode=(0.05, 0, 0))  # Explode first slice
        
        # Style the text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        axes[1].set_title('Economic Impact Distribution\nAnnual Costs by Category', 
                         fontweight='bold')
        
        # Add center text
        axes[1].text(0, 0, '€2.37B\nTotal', ha='center', va='center', 
                    fontweight='bold', fontsize=16,
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self._save_figure('economic_impact_comparison')
        plt.close()
        
        print("  → Impact comparison created")
    
    def create_cost_per_vehicle_analysis(self):
        """Create cost per vehicle and per capita analysis"""
        print("Creating cost per vehicle analysis...")
        
        # Data assumptions
        population_slovenia = 2100000  # 2.1 million
        annual_vehicle_km = 18.5e9  # 18.5 billion vehicle-km
        daily_vehicles = 300000  # Average daily vehicles on highways
        
        # Calculate per-unit costs
        total_annual_cost = 2370  # Million euros
        cost_per_capita = total_annual_cost * 1e6 / population_slovenia
        cost_per_vehicle_km = total_annual_cost * 1e6 / annual_vehicle_km
        cost_per_vehicle_day = total_annual_cost * 1e6 / (daily_vehicles * 365)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cost per capita by category
        categories = ['Congestion\nDelays', 'Accidents', 'Fuel', 'Emissions', 'Other']
        costs_m = [287, 142, 43, 76, 552]  # Million euros
        per_capita_costs = [cost * 1e6 / population_slovenia for cost in costs_m]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars = axes[0,0].bar(categories, per_capita_costs, color=colors, 
                            alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, cost in zip(bars, per_capita_costs):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 10,
                          f'€{cost:.0f}', ha='center', va='bottom', fontweight='bold')
        
        axes[0,0].set_ylabel('Cost per Capita (€/person/year)')
        axes[0,0].set_title('Annual Traffic Costs per Capita by Category', fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # 2. Cost per vehicle-km
        axes[0,1].bar(['Current System'], [cost_per_vehicle_km * 1000], 
                     color='red', alpha=0.8, width=0.5)
        axes[0,1].set_ylabel('Cost per Vehicle-km (€ cents)')
        axes[0,1].set_title('Traffic Cost per Vehicle-Kilometer\nCurrent Network Performance', 
                           fontweight='bold')
        axes[0,1].text(0, cost_per_vehicle_km * 1000 + 0.2, 
                      f'{cost_per_vehicle_km * 1000:.1f}¢', 
                      ha='center', va='bottom', fontweight='bold', fontsize=14)
        axes[0,1].grid(True, alpha=0.3, axis='y')
        
        # 3. International comparison (estimated)
        countries = ['Slovenia\n(Current)', 'Germany', 'France', 'Netherlands', 'Austria']
        congestion_costs_per_capita = [cost_per_capita, 950, 1200, 1400, 800]  # Estimated
        
        bars = axes[1,0].bar(countries, congestion_costs_per_capita, 
                            color=['red', 'blue', 'blue', 'blue', 'blue'], 
                            alpha=0.7)
        
        axes[1,0].set_ylabel('Annual Cost per Capita (€)')
        axes[1,0].set_title('Traffic Congestion Costs: International Comparison\n(Estimated)', 
                           fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # 4. Cost breakdown pie chart
        cost_types = ['Time Value\n(Delays)', 'Fuel Costs', 'Vehicle Wear', 
                     'Emissions', 'Accidents', 'Other']
        cost_shares = [45, 15, 10, 12, 8, 10]  # Percentages
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        wedges, texts, autotexts = axes[1,1].pie(cost_shares, labels=cost_types, 
                                                colors=colors_pie, autopct='%1.1f%%')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        axes[1,1].set_title('Cost Composition Breakdown\nBy Economic Impact Type', 
                           fontweight='bold')
        
        plt.tight_layout()
        self._save_figure('cost_per_vehicle_analysis')
        plt.close()
        
        print("  → Cost per vehicle analysis created")
        
        # Print summary
        print(f"    Cost per capita: €{cost_per_capita:.0f}/year")
        print(f"    Cost per vehicle-km: {cost_per_vehicle_km*1000:.1f} cents")
    
    def _save_figure(self, name):
        """Save figure in publication format"""
        self.figure_count += 1
        filename = f"fig_{self.figure_count:02d}_{name}"
        
        # Save as PDF
        pdf_path = self.figures_dir / f"{filename}.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        
        # Save as PNG
        png_path = self.figures_dir / f"{filename}.png"
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        
        print(f"    → Saved: {filename}")

def main():
    """Extract economic impact figures"""
    print("💰 ECONOMIC IMPACT FIGURE EXTRACTION")
    print("="*50)
    
    extractor = EconomicFigureExtractor()
    
    # Create economic analysis figures
    extractor.create_cost_breakdown_waterfall()
    extractor.create_benefit_cost_analysis()
    extractor.create_impact_comparison()
    extractor.create_cost_per_vehicle_analysis()
    
    print("\n✅ Economic impact figures extracted successfully!")
    print("Key metrics: BCR=4.8, IRR=72%, Payback=1.4 years")

if __name__ == "__main__":
    main()