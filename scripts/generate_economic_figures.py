#!/usr/bin/env python3
"""
Generate economic analysis figures for the arXiv article.
Task 3.4: Economic Analysis Figures

This script generates three publication-quality figures:
1. ROI timeline showing cumulative NPV over 10 years
2. Cost-benefit waterfall chart showing €2.37B annual impact breakdown
3. NPV sensitivity analysis heatmap

Data source: Notebook 12 - Economic Impact Assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import json

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'article' / 'figures'

# Create figures directory if it doesn't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'text.usetex': False
})

def load_economic_data():
    """Load economic impact results from JSON file."""
    results_file = REPORTS_DIR / 'economic_impact_results.json'
    with open(results_file, 'r') as f:
        return json.load(f)

def generate_roi_timeline():
    """
    Generate Figure 12: ROI Timeline showing cumulative NPV over 10 years.
    Shows investment returns for each intervention with break-even points.
    """
    print("Generating Figure 12: ROI timeline...")
    
    # Load economic data
    economic_data = load_economic_data()
    
    # Define interventions with detailed financial data
    interventions = {
        'Real-time Information': {
            'initial_cost': 500_000,
            'annual_cost': 50_000,
            'annual_benefit': 1_000_000,
            'color': '#2E7D32',  # Dark green
            'style': '-'
        },
        'AI Traffic Management': {
            'initial_cost': 5_000_000,
            'annual_cost': 500_000,
            'annual_benefit': 8_000_000,
            'color': '#1976D2',  # Blue
            'style': '-'
        },
        'Variable Speed Limits': {
            'initial_cost': 2_000_000,
            'annual_cost': 200_000,
            'annual_benefit': 3_000_000,
            'color': '#F57C00',  # Orange
            'style': '--'
        },
        'Early Warning System': {
            'initial_cost': 1_000_000,
            'annual_cost': 100_000,
            'annual_benefit': 1_500_000,
            'color': '#7B1FA2',  # Purple
            'style': '--'
        },
        'Ramp Metering': {
            'initial_cost': 3_000_000,
            'annual_cost': 300_000,
            'annual_benefit': 4_000_000,
            'color': '#C62828',  # Red
            'style': '-.'
        }
    }
    
    # Calculate NPV over 10 years
    discount_rate = 0.03
    years = np.arange(0, 11)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Individual intervention NPVs
    for name, params in interventions.items():
        npv_values = []
        for year in years:
            if year == 0:
                npv = -params['initial_cost']
            else:
                annual_net = params['annual_benefit'] - params['annual_cost']
                npv = -params['initial_cost'] + sum([annual_net / (1 + discount_rate)**t 
                                                     for t in range(1, year+1)])
            npv_values.append(npv / 1e6)  # Convert to millions
        
        ax1.plot(years, npv_values, linewidth=2.5, label=name[:15], 
                color=params['color'], linestyle=params['style'],
                marker='o', markersize=4, markevery=2)
    
    # Add break-even line
    ax1.axhline(y=0, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Break-even')
    ax1.fill_between(years, -10, 0, alpha=0.1, color='red')
    ax1.fill_between(years, 0, 40, alpha=0.1, color='green')
    
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Cumulative NPV (Million €)', fontsize=11)
    ax1.set_title('Individual Intervention NPV Trajectories', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-8, 35)
    
    # Add annotations for key metrics
    ax1.text(0.98, 0.02, 'Discount rate: 3%', transform=ax1.transAxes,
            fontsize=8, ha='right', va='bottom', style='italic')
    
    # Subplot 2: Portfolio comparison
    scenarios = {
        'Do Nothing': {'color': '#B71C1C', 'cumulative_cost': []},
        'Quick Wins Only': {'color': '#FF6F00', 'cumulative_cost': []},
        'Full Portfolio': {'color': '#1B5E20', 'cumulative_cost': []}
    }
    
    # Calculate scenario costs
    base_annual_cost = economic_data['total_annual_impact'] / 1e6  # €2.37B in millions
    
    for year in years:
        # Do Nothing - costs increase 5% annually
        scenarios['Do Nothing']['cumulative_cost'].append(
            sum([base_annual_cost * (1.05)**t for t in range(year+1)])
        )
        
        # Quick Wins - 15% reduction after year 1
        if year == 0:
            scenarios['Quick Wins Only']['cumulative_cost'].append(base_annual_cost)
        else:
            scenarios['Quick Wins Only']['cumulative_cost'].append(
                base_annual_cost + sum([base_annual_cost * 0.85 * (1.02)**t for t in range(year)])
            )
        
        # Full Portfolio - 40% reduction after year 2
        if year <= 1:
            scenarios['Full Portfolio']['cumulative_cost'].append(
                base_annual_cost * (year + 1)
            )
        else:
            scenarios['Full Portfolio']['cumulative_cost'].append(
                base_annual_cost * 2 + sum([base_annual_cost * 0.6 * (0.98)**t for t in range(year-1)])
            )
    
    # Plot scenarios
    for name, data in scenarios.items():
        ax2.plot(years, data['cumulative_cost'], linewidth=3, 
                label=name, color=data['color'], marker='s', markersize=5, markevery=2)
    
    # Add savings annotation
    savings_10yr = scenarios['Do Nothing']['cumulative_cost'][-1] - scenarios['Full Portfolio']['cumulative_cost'][-1]
    ax2.annotate(f'10-year savings:\n€{savings_10yr:.0f}M', 
                xy=(10, scenarios['Full Portfolio']['cumulative_cost'][-1]),
                xytext=(8, scenarios['Full Portfolio']['cumulative_cost'][-1] - 5000),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Cumulative Economic Cost (Million €)', fontsize=11)
    ax2.set_title('Scenario Comparison: 10-Year Economic Impact', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_xlim(-0.5, 10.5)
    
    # Add text box with key findings
    textstr = f'Current annual impact: €{base_annual_cost:.0f}M\\n10-year savings potential: €{economic_data["optimization_potential"]["ten_year_savings"]/1e9:.1f}B\\nPayback period: {economic_data["optimization_potential"]["payback_months"]:.1f} months'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.suptitle('Economic Return on Investment Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'fig_12_roi_timeline.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def generate_cost_benefit_waterfall():
    """
    Generate Figure 13: Cost-benefit waterfall chart.
    Shows the breakdown of €2.37B annual economic impact and potential savings.
    """
    print("Generating Figure 13: Cost-benefit waterfall...")
    
    # Load economic data
    economic_data = load_economic_data()
    
    # Prepare data for waterfall chart
    categories = ['Direct\nCosts', 'Indirect\nCosts', 'Environmental\nCosts', 'Social\nCosts', 
                  'Total Impact', 'Quick Wins\n(-15%)', 'Optimization\n(-25%)', 'Full Program\n(-40%)', 
                  'Net Impact']
    
    # Values in millions
    direct = economic_data['cost_breakdown']['direct'] / 1e6
    indirect = economic_data['cost_breakdown']['indirect'] / 1e6
    environmental = economic_data['cost_breakdown']['environmental'] / 1e6
    social = economic_data['cost_breakdown']['social'] / 1e6
    total = economic_data['total_annual_impact'] / 1e6
    
    # Calculate reductions
    quick_wins = -total * 0.15
    optimization = -total * 0.25
    full_program = -total * 0.40
    net_impact = total + quick_wins + optimization + full_program
    
    values = [direct, indirect, environmental, social, 0, quick_wins, optimization, full_program, 0]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Waterfall chart
    # Calculate positions
    cumulative = 0
    bar_starts = []
    bar_values = []
    colors = []
    
    for i, val in enumerate(values):
        if i == 4:  # Total Impact bar
            bar_starts.append(0)
            bar_values.append(total)
            colors.append('#B71C1C')  # Dark red for total
        elif i == 8:  # Net Impact bar
            bar_starts.append(0)
            bar_values.append(net_impact)
            colors.append('#1B5E20')  # Dark green for net
        else:
            if i < 4:  # Cost components
                bar_starts.append(cumulative)
                bar_values.append(val)
                cumulative += val
                colors.append('#FF5252' if i == 0 else '#FF9800' if i == 1 else '#FFC107' if i == 2 else '#FFEB3B')
            else:  # Reductions
                bar_starts.append(cumulative + val if val < 0 else cumulative)
                bar_values.append(abs(val))
                cumulative += val
                colors.append('#4CAF50' if i == 5 else '#66BB6A' if i == 6 else '#81C784')
    
    # Create bars
    bars = ax1.bar(range(len(categories)), bar_values, bottom=bar_starts, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add connecting lines
    for i in range(len(categories) - 1):
        if i not in [3, 4, 7, 8]:  # Skip lines for total and net bars
            if i < 3:
                y_start = bar_starts[i] + bar_values[i]
                y_end = bar_starts[i+1]
            else:
                y_start = bar_starts[i]
                y_end = bar_starts[i+1] + bar_values[i+1] if values[i+1] < 0 else bar_starts[i+1]
            
            ax1.plot([i + 0.4, i + 0.6], [y_start, y_end], 'k--', alpha=0.5, linewidth=1)
    
    # Add value labels
    for i, (start, height, cat) in enumerate(zip(bar_starts, bar_values, categories)):
        if height > 50:  # Only label significant bars
            ax1.text(i, start + height/2, f'€{height:.0f}M', 
                    ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Formatting
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=0, ha='center')
    ax1.set_ylabel('Annual Economic Impact (Million €)', fontsize=11)
    ax1.set_title('Cost Components and Optimization Impact', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(total, net_impact) * 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    cost_patch = mpatches.Patch(color='#FF5252', label='Cost Components', alpha=0.8)
    reduction_patch = mpatches.Patch(color='#4CAF50', label='Cost Reductions', alpha=0.8)
    ax1.legend(handles=[cost_patch, reduction_patch], loc='upper right')
    
    # Subplot 2: Cost breakdown pie chart with interventions
    fig2_data = {
        'Direct Costs': direct,
        'Indirect Costs': indirect,
        'Environmental': environmental,
        'Social Costs': social
    }
    
    # Create nested pie chart
    outer_values = list(fig2_data.values())
    outer_labels = list(fig2_data.keys())
    outer_colors = ['#FF5252', '#FF9800', '#FFC107', '#FFEB3B']
    
    # Inner pie for interventions impact
    inner_values = [total * 0.6, total * 0.4]  # 60% remaining, 40% saved
    inner_labels = ['Remaining Impact', 'Achievable Savings']
    inner_colors = ['#FFCDD2', '#C8E6C9']
    
    # Create pie charts
    wedges1, texts1, autotexts1 = ax2.pie(outer_values, labels=outer_labels, colors=outer_colors,
                                           autopct='%1.1f%%', startangle=90, radius=1,
                                           textprops={'fontsize': 9}, pctdistance=0.85)
    
    wedges2, texts2, autotexts2 = ax2.pie(inner_values, labels=None, colors=inner_colors,
                                           autopct='', startangle=90, radius=0.6)
    
    # Add center text
    ax2.text(0, 0, f'€{total:.0f}M\nAnnual Impact', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    ax2.set_title('Economic Impact Distribution', fontsize=12, fontweight='bold')
    
    # Add annotations
    ax2.annotate('40% reduction\nachievable', xy=(0.3, -0.3), xytext=(0.7, -0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=9, color='green', fontweight='bold')
    
    plt.suptitle('Economic Impact Analysis: €2.37B Annual Cost Breakdown', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'fig_13_cost_benefit_waterfall.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def generate_npv_sensitivity():
    """
    Generate Figure 14: NPV sensitivity analysis heatmap.
    Shows how NPV changes with different parameter variations.
    """
    print("Generating Figure 14: NPV sensitivity analysis...")
    
    # Load economic data
    economic_data = load_economic_data()
    
    # Define parameter ranges for sensitivity analysis
    vot_range = np.linspace(-30, 30, 15)  # Value of Time variation (%)
    traffic_growth_range = np.linspace(-20, 20, 15)  # Traffic growth variation (%)
    
    # Base NPV (10-year savings in billions)
    base_npv = economic_data['optimization_potential']['ten_year_savings'] / 1e9
    
    # Create sensitivity matrix
    sensitivity_matrix = np.zeros((len(traffic_growth_range), len(vot_range)))
    
    for i, traffic_var in enumerate(traffic_growth_range):
        for j, vot_var in enumerate(vot_range):
            # Calculate NPV impact
            # VoT affects 60% of costs, traffic affects 80%
            vot_impact = base_npv * (1 + vot_var/100 * 0.6)
            traffic_impact = vot_impact * (1 + traffic_var/100 * 0.8)
            sensitivity_matrix[i, j] = traffic_impact
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Subplot 1: Main sensitivity heatmap
    ax1 = axes[0, 0]
    im = ax1.imshow(sensitivity_matrix, aspect='auto', cmap='RdYlGn', 
                    vmin=0, vmax=base_npv*2, origin='lower')
    
    # Add contour lines
    contours = ax1.contour(sensitivity_matrix, levels=[5, 10, 15, 20, 25, 30], 
                          colors='black', alpha=0.4, linewidths=1)
    ax1.clabel(contours, inline=True, fontsize=8, fmt='€%.0fB')
    
    # Mark base case
    base_i = len(traffic_growth_range) // 2
    base_j = len(vot_range) // 2
    ax1.plot(base_j, base_i, 'k*', markersize=15, label='Base Case')
    
    # Set labels
    ax1.set_xlabel('Value of Time Change (%)', fontsize=10)
    ax1.set_ylabel('Traffic Growth Change (%)', fontsize=10)
    ax1.set_title('10-Year NPV Sensitivity (€ Billions)', fontsize=11, fontweight='bold')
    
    # Set tick labels
    x_ticks = np.linspace(0, len(vot_range)-1, 7, dtype=int)
    y_ticks = np.linspace(0, len(traffic_growth_range)-1, 7, dtype=int)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{vot_range[i]:.0f}' for i in x_ticks])
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([f'{traffic_growth_range[i]:.0f}' for i in y_ticks])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, label='NPV (€ Billions)')
    
    # Subplot 2: Tornado diagram for single-factor sensitivity
    ax2 = axes[0, 1]
    
    parameters = {
        'Value of Time': {'low': -30, 'high': 30, 'impact': 0.6},
        'Traffic Volume': {'low': -20, 'high': 20, 'impact': 0.8},
        'Fuel Prices': {'low': -50, 'high': 50, 'impact': 0.15},
        'CO2 Prices': {'low': -50, 'high': 100, 'impact': 0.05},
        'Discount Rate': {'low': -1, 'high': 2, 'impact': 0.3},
        'Implementation Cost': {'low': -20, 'high': 30, 'impact': -0.2}
    }
    
    # Calculate impacts
    tornado_data = []
    for param, values in parameters.items():
        low_impact = base_npv * (1 + values['low']/100 * values['impact'])
        high_impact = base_npv * (1 + values['high']/100 * values['impact'])
        tornado_data.append({
            'parameter': param,
            'low': low_impact,
            'high': high_impact,
            'range': abs(high_impact - low_impact)
        })
    
    # Sort by range
    tornado_data.sort(key=lambda x: x['range'], reverse=True)
    
    # Plot tornado
    y_pos = np.arange(len(tornado_data))
    for i, data in enumerate(tornado_data):
        low = min(data['low'], data['high'])
        width = abs(data['high'] - data['low'])
        bar = ax2.barh(i, width, left=low, height=0.7,
                      color='steelblue' if data['high'] > data['low'] else 'coral',
                      alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels
        ax2.text(data['low'], i, f'{data["low"]:.1f}', ha='right', va='center', fontsize=8)
        ax2.text(data['high'], i, f'{data["high"]:.1f}', ha='left', va='center', fontsize=8)
    
    # Add base line
    ax2.axvline(x=base_npv, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Base NPV')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([d['parameter'] for d in tornado_data])
    ax2.set_xlabel('10-Year NPV (€ Billions)', fontsize=10)
    ax2.set_title('Single-Factor Sensitivity Analysis', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Subplot 3: BCR sensitivity to discount rate
    ax3 = axes[1, 0]
    
    discount_rates = np.linspace(0, 0.1, 50)
    bcr_values = []
    
    for rate in discount_rates:
        # Calculate BCR with different discount rates
        total_benefits = sum([26.1e6 / (1 + rate)**t for t in range(1, 11)])
        total_costs = 26.9e6  # Initial investment
        bcr = total_benefits / total_costs if total_costs > 0 else 0
        bcr_values.append(bcr)
    
    ax3.plot(discount_rates * 100, bcr_values, linewidth=3, color='darkblue')
    ax3.fill_between(discount_rates * 100, bcr_values, alpha=0.3, color='lightblue')
    
    # Mark current rate
    current_rate = 0.03
    current_bcr = bcr_values[np.argmin(np.abs(discount_rates - current_rate))]
    ax3.plot(current_rate * 100, current_bcr, 'ro', markersize=10, label=f'Current: {current_bcr:.2f}')
    
    # Add reference lines
    ax3.axhline(y=1, color='red', linestyle=':', alpha=0.5, label='Break-even (BCR=1)')
    ax3.axhline(y=3, color='green', linestyle=':', alpha=0.5, label='High return (BCR=3)')
    
    ax3.set_xlabel('Discount Rate (%)', fontsize=10)
    ax3.set_ylabel('Benefit-Cost Ratio', fontsize=10)
    ax3.set_title('BCR Sensitivity to Discount Rate', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)
    
    # Subplot 4: Monte Carlo simulation results
    ax4 = axes[1, 1]
    
    # Simulate NPV distribution
    np.random.seed(42)
    n_simulations = 10000
    
    # Generate random variations
    vot_variations = np.random.normal(0, 10, n_simulations)  # ±10% std dev
    traffic_variations = np.random.normal(0, 5, n_simulations)  # ±5% std dev
    cost_variations = np.random.normal(0, 15, n_simulations)  # ±15% std dev
    
    # Calculate NPV distribution
    npv_distribution = []
    for vot, traffic, cost in zip(vot_variations, traffic_variations, cost_variations):
        npv = base_npv * (1 + vot/100 * 0.6) * (1 + traffic/100 * 0.8) * (1 - cost/100 * 0.2)
        npv_distribution.append(npv)
    
    # Plot histogram
    counts, bins, patches = ax4.hist(npv_distribution, bins=50, density=True, 
                                     alpha=0.7, color='skyblue', edgecolor='black')
    
    # Fit and plot normal distribution
    mu, std = np.mean(npv_distribution), np.std(npv_distribution)
    x = np.linspace(min(npv_distribution), max(npv_distribution), 100)
    ax4.plot(x, 1/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std)**2),
            'r-', linewidth=2, label='Normal fit')
    
    # Add percentile lines
    percentiles = [5, 50, 95]
    colors_p = ['red', 'black', 'green']
    for p, c in zip(percentiles, colors_p):
        val = np.percentile(npv_distribution, p)
        ax4.axvline(x=val, color=c, linestyle='--', alpha=0.7, 
                   label=f'{p}th percentile: €{val:.1f}B')
    
    ax4.set_xlabel('10-Year NPV (€ Billions)', fontsize=10)
    ax4.set_ylabel('Probability Density', fontsize=10)
    ax4.set_title(f'Monte Carlo NPV Distribution ({n_simulations:,} simulations)', 
                 fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Mean: €{mu:.1f}B\nStd Dev: €{std:.1f}B\nP(NPV>€15B): {sum(1 for x in npv_distribution if x > 15)/n_simulations*100:.1f}%'
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Sensitivity and Uncertainty Analysis of Economic Projections', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'fig_14_npv_sensitivity.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    """Generate all economic analysis figures."""
    print("\n" + "="*60)
    print("Generating Economic Analysis Figures")
    print("Task 3.4: Economic Analysis Figures")
    print("="*60 + "\n")
    
    # Generate figures
    generate_roi_timeline()
    generate_cost_benefit_waterfall()
    generate_npv_sensitivity()
    
    print("\n" + "="*60)
    print("Successfully generated all economic analysis figures!")
    print("Output directory:", FIGURES_DIR)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()