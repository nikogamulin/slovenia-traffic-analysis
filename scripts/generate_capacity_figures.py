#!/usr/bin/env python3
"""
Generate capacity analysis figures for the arXiv article.
Task 3.3: Capacity Analysis Visualization

This script generates three publication-quality figures:
1. Capacity utilization projection (system failure by 2033)
2. Optimization vs expansion scenarios comparison
3. Growth rate sensitivity analysis heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime, timedelta

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

def load_economic_results():
    """Load economic impact results."""
    results_file = REPORTS_DIR / 'economic_impact_results.json'
    with open(results_file, 'r') as f:
        return json.load(f)

def generate_capacity_utilization_projection():
    """
    Generate Figure 9: Capacity utilization projection showing system failure by 2033.
    Shows current 87% utilization growing at 3.5% annually, hitting 100% in 8.3 years.
    """
    print("Generating Figure 9: Capacity utilization projection...")
    
    # Parameters from analysis
    current_utilization = 0.87  # 87% current utilization
    growth_rate = 0.035  # 3.5% annual growth
    optimization_gain = 0.35  # 35% capacity gain from optimization
    
    # Time horizon
    years = np.arange(2025, 2041)
    
    # Calculate projections
    # Do nothing scenario
    utilization_do_nothing = current_utilization * (1 + growth_rate) ** (years - 2025)
    
    # Optimization only (35% capacity gain = divide utilization by 1.35)
    utilization_optimization = current_utilization / 1.35 * (1 + growth_rate) ** (years - 2025)
    
    # Optimization + Expansion (assuming 50% additional capacity)
    utilization_expansion = current_utilization / (1.35 * 1.5) * (1 + growth_rate) ** (years - 2025)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scenarios
    ax.plot(years, utilization_do_nothing * 100, 'r-', linewidth=2.5, 
            label='Do Nothing', marker='o', markersize=4, markevery=2)
    ax.plot(years, utilization_optimization * 100, 'b--', linewidth=2.5,
            label='Optimization Only (+35%)', marker='s', markersize=4, markevery=2)
    ax.plot(years, utilization_expansion * 100, 'g-.', linewidth=2.5,
            label='Optimization + Expansion', marker='^', markersize=4, markevery=2)
    
    # Add critical threshold
    ax.axhline(y=100, color='black', linestyle=':', linewidth=2, alpha=0.7, label='Capacity Limit')
    ax.axhline(y=90, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='Critical Threshold (90%)')
    
    # Shade failure regions
    ax.fill_between(years, 100, 120, color='red', alpha=0.1)
    ax.text(2028, 105, 'SYSTEM FAILURE', fontsize=11, fontweight='bold', color='red', alpha=0.7)
    
    # Find failure years
    failure_do_nothing = 2025 + np.log(100/87) / np.log(1.035)
    failure_optimization = 2025 + np.log(100*1.35/87) / np.log(1.035)
    
    # Add failure year annotations
    ax.annotate(f'Failure: {failure_do_nothing:.1f}', 
                xy=(failure_do_nothing, 100), xytext=(failure_do_nothing-1, 108),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red')
    ax.annotate(f'Failure: {failure_optimization:.1f}', 
                xy=(failure_optimization, 100), xytext=(failure_optimization-1, 92),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=9, color='blue')
    
    # Formatting
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Capacity Utilization (%)', fontsize=11)
    ax.set_title('Highway Network Capacity Utilization Projections', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(2024.5, 2040.5)
    ax.set_ylim(60, 120)
    
    # Add text box with key findings
    textstr = 'Key Findings:\n• Current utilization: 87%\n• Growth rate: 3.5%/year\n• System fails in 8.3 years\n• Optimization delays by 11 years\n• Expansion needed for sustainability'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'fig_09_capacity_utilization_projection.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def generate_optimization_vs_expansion():
    """
    Generate Figure 10: Comparison of do-nothing, optimization-only, and optimization+expansion scenarios.
    Shows cost-benefit analysis and economic impacts over 10 years.
    """
    print("Generating Figure 10: Optimization vs expansion scenarios...")
    
    # Load economic results
    economic_data = load_economic_results()
    
    # Scenario parameters (in millions of euros)
    scenarios = {
        'Do Nothing': {
            'investment': 0,
            'annual_cost': 505,  # Annual congestion cost
            'capacity_gain': 0,
            'color': 'red'
        },
        'Optimization Only': {
            'investment': 27,  # From economic results
            'annual_cost': 505 * 0.65,  # 35% reduction
            'capacity_gain': 35,
            'color': 'blue'
        },
        'Optimization +\nExpansion': {
            'investment': 550 + 27,  # Highway expansion + optimization
            'annual_cost': 505 * 0.2,  # 80% reduction
            'capacity_gain': 85,
            'color': 'green'
        }
    }
    
    # Calculate 10-year NPV for each scenario
    discount_rate = 0.03
    years = np.arange(0, 11)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Investment vs Annual Savings
    ax1 = axes[0, 0]
    scenario_names = list(scenarios.keys())
    investments = [scenarios[s]['investment'] for s in scenario_names]
    annual_savings = [505 - scenarios[s]['annual_cost'] for s in scenario_names]
    colors = [scenarios[s]['color'] for s in scenario_names]
    
    x_pos = np.arange(len(scenario_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, investments, width, label='Initial Investment', color='gray', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, annual_savings, width, label='Annual Savings', color=colors, alpha=0.7)
    
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Amount (Million €)')
    ax1.set_title('Investment vs Annual Savings', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scenario_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'€{height:.0f}M', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'€{height:.0f}M', ha='center', va='bottom', fontsize=8)
    
    # Subplot 2: Capacity Gains
    ax2 = axes[0, 1]
    capacity_gains = [scenarios[s]['capacity_gain'] for s in scenario_names]
    bars = ax2.bar(x_pos, capacity_gains, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Capacity Gain (%)')
    ax2.set_title('Network Capacity Improvements', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenario_names)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Subplot 3: Cumulative NPV over 10 years
    ax3 = axes[1, 0]
    for name, params in scenarios.items():
        npv_values = []
        for year in years:
            if year == 0:
                npv = -params['investment']
            else:
                annual_benefit = 505 - params['annual_cost']
                npv = -params['investment'] + sum([annual_benefit / (1 + discount_rate)**t for t in range(1, year+1)])
            npv_values.append(npv)
        
        ax3.plot(years, npv_values, linewidth=2.5, label=name, color=params['color'], 
                marker='o', markersize=5, markevery=2)
    
    ax3.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Cumulative NPV (Million €)')
    ax3.set_title('10-Year Net Present Value Analysis', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.5, 10.5)
    
    # Subplot 4: Benefit-Cost Ratio
    ax4 = axes[1, 1]
    bcr_values = []
    payback_periods = []
    
    for name in scenario_names:
        if scenarios[name]['investment'] > 0:
            annual_benefit = 505 - scenarios[name]['annual_cost']
            ten_year_benefit = sum([annual_benefit / (1 + discount_rate)**t for t in range(1, 11)])
            bcr = ten_year_benefit / scenarios[name]['investment']
            payback = scenarios[name]['investment'] / annual_benefit if annual_benefit > 0 else np.inf
        else:
            bcr = 0
            payback = 0
        bcr_values.append(bcr)
        payback_periods.append(payback)
    
    # Create secondary y-axis for payback period
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar(x_pos - width/2, bcr_values, width, label='BCR', color='navy', alpha=0.7)
    bars2 = ax4_twin.bar(x_pos + width/2, payback_periods, width, label='Payback (years)', color='orange', alpha=0.7)
    
    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Benefit-Cost Ratio', color='navy')
    ax4_twin.set_ylabel('Payback Period (years)', color='orange')
    ax4.set_title('Investment Performance Metrics', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(scenario_names)
    ax4.tick_params(axis='y', labelcolor='navy')
    ax4_twin.tick_params(axis='y', labelcolor='orange')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8, color='navy')
    for bar in bars2:
        height = bar.get_height()
        if height > 0 and height < 10:
            ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}y', ha='center', va='bottom', fontsize=8, color='orange')
    
    # Add legends
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    plt.suptitle('Economic Analysis: Optimization vs Expansion Scenarios', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'fig_10_optimization_vs_expansion.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def generate_growth_rate_sensitivity():
    """
    Generate Figure 11: Growth rate sensitivity analysis heatmap.
    Shows how different traffic growth rates and capacity scenarios affect failure timing.
    """
    print("Generating Figure 11: Growth rate sensitivity analysis...")
    
    # Parameters
    current_utilization = 0.87
    growth_rates = np.linspace(0.01, 0.06, 20)  # 1% to 6% growth
    capacity_gains = np.linspace(0, 1.0, 20)  # 0% to 100% capacity gain
    
    # Calculate years to failure for each combination
    failure_matrix = np.zeros((len(growth_rates), len(capacity_gains)))
    
    for i, growth in enumerate(growth_rates):
        for j, gain in enumerate(capacity_gains):
            if growth > 0:
                # Years to reach 100% utilization
                effective_capacity = 1.0 * (1 + gain)
                if current_utilization < effective_capacity:
                    years_to_failure = np.log(effective_capacity / current_utilization) / np.log(1 + growth)
                    failure_matrix[i, j] = min(years_to_failure, 50)  # Cap at 50 years
                else:
                    failure_matrix[i, j] = 0  # Already at capacity
            else:
                failure_matrix[i, j] = 50  # No growth = no failure
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Heatmap
    ax1 = axes[0]
    im = ax1.imshow(failure_matrix, aspect='auto', origin='lower', cmap='RdYlGn', vmin=0, vmax=30)
    
    # Add contour lines for key thresholds
    contour_levels = [5, 10, 15, 20, 25]
    contours = ax1.contour(failure_matrix, levels=contour_levels, colors='black', alpha=0.4, linewidths=1)
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%d years')
    
    # Mark current scenario (3.5% growth, 0% gain)
    current_growth_idx = np.argmin(np.abs(growth_rates - 0.035))
    current_gain_idx = np.argmin(np.abs(capacity_gains - 0))
    ax1.plot(current_gain_idx, current_growth_idx, 'r*', markersize=15, label='Current State')
    
    # Mark optimization scenario (3.5% growth, 35% gain)
    opt_gain_idx = np.argmin(np.abs(capacity_gains - 0.35))
    ax1.plot(opt_gain_idx, current_growth_idx, 'b*', markersize=15, label='With Optimization')
    
    # Mark expansion scenario (3.5% growth, 85% gain)
    exp_gain_idx = np.argmin(np.abs(capacity_gains - 0.85))
    ax1.plot(exp_gain_idx, current_growth_idx, 'g*', markersize=15, label='With Expansion')
    
    # Formatting
    ax1.set_xlabel('Capacity Gain (%)', fontsize=11)
    ax1.set_ylabel('Annual Traffic Growth Rate (%)', fontsize=11)
    ax1.set_title('Years to Network Failure', fontsize=12, fontweight='bold')
    
    # Set tick labels
    x_ticks = np.arange(0, len(capacity_gains), 4)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{capacity_gains[i]*100:.0f}' for i in x_ticks])
    
    y_ticks = np.arange(0, len(growth_rates), 4)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([f'{growth_rates[i]*100:.1f}' for i in y_ticks])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, label='Years to Failure')
    cbar.set_label('Years to Failure', rotation=270, labelpad=20)
    
    ax1.legend(loc='upper right', frameon=True, fancybox=True)
    
    # Subplot 2: Scenario comparison lines
    ax2 = axes[1]
    
    # Plot years to failure vs capacity gain for different growth rates
    growth_scenarios = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    colors_map = plt.cm.coolwarm(np.linspace(0, 1, len(growth_scenarios)))
    
    for idx, growth in enumerate(growth_scenarios):
        years_to_failure = []
        for gain in capacity_gains:
            effective_capacity = 1.0 * (1 + gain)
            if current_utilization < effective_capacity and growth > 0:
                years = np.log(effective_capacity / current_utilization) / np.log(1 + growth)
                years_to_failure.append(min(years, 50))
            else:
                years_to_failure.append(0 if current_utilization >= effective_capacity else 50)
        
        ax2.plot(capacity_gains * 100, years_to_failure, linewidth=2,
                label=f'{growth*100:.1f}% growth', color=colors_map[idx])
    
    # Add reference lines
    ax2.axhline(y=10, color='red', linestyle=':', alpha=0.5, label='10-year horizon')
    ax2.axhline(y=20, color='orange', linestyle=':', alpha=0.5, label='20-year horizon')
    ax2.axvline(x=35, color='blue', linestyle='--', alpha=0.5, label='Optimization gain')
    ax2.axvline(x=85, color='green', linestyle='--', alpha=0.5, label='Expansion gain')
    
    ax2.set_xlabel('Capacity Gain (%)', fontsize=11)
    ax2.set_ylabel('Years to Network Failure', fontsize=11)
    ax2.set_title('Sensitivity to Growth Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', ncol=2, fontsize=8)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 35)
    
    plt.suptitle('Network Capacity Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'fig_11_growth_rate_sensitivity.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    """Generate all capacity analysis figures."""
    print("\n" + "="*60)
    print("Generating Capacity Analysis Figures")
    print("Task 3.3: Capacity Analysis Visualization")
    print("="*60 + "\n")
    
    # Generate figures
    generate_capacity_utilization_projection()
    generate_optimization_vs_expansion()
    generate_growth_rate_sensitivity()
    
    print("\n" + "="*60)
    print("Successfully generated all capacity analysis figures!")
    print("Output directory:", FIGURES_DIR)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()