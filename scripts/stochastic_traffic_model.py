#!/usr/bin/env python3
"""
Stochastic Traffic Model with Monte Carlo Simulation
Task 4.2: Account for uncertainty in traffic projections

This script implements a stochastic model to quantify uncertainty in traffic 
growth projections and system failure timing. Uses Monte Carlo simulation with
10,000 iterations to generate probability distributions.

Key stochastic components:
- Growth rate uncertainty: g_t ~ N(0.035, 0.01²)
- Demand variability: D_t ~ Poisson(λ_t)
- Capacity uncertainty: C ~ N(C_nom, σ_C²)

Output:
- Probability distribution of failure year
- Cumulative failure probability curves
- Confidence intervals and sensitivity analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'article' / 'figures'

# Create directories if they don't exist
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

# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

class ModelParameters:
    """Central repository for all model parameters with uncertainty."""
    
    # Base parameters (deterministic values)
    C_0 = 6600  # Current capacity (veh/hr)
    U_0 = 0.87  # Current utilization (87%)
    U_c = 0.95  # Critical utilization threshold
    ALPHA_OPT = 0.35  # Maximum optimization gain
    
    # Stochastic parameters (mean, std)
    GROWTH_MEAN = 0.035  # Mean annual growth rate
    GROWTH_STD = 0.010   # Std dev of growth rate
    
    # Capacity uncertainty
    CAPACITY_STD_RATIO = 0.05  # 5% of nominal capacity
    
    # Demand variability (Poisson parameter modifiers)
    PEAK_HOUR_MULTIPLIER = 1.8
    OFF_PEAK_MULTIPLIER = 0.6
    WEEKEND_MULTIPLIER = 0.7
    
    # Time parameters
    YEARS_TO_SIMULATE = 20
    IMPLEMENTATION_DELAY = 3  # Years to implement optimization
    
    # Simulation parameters
    N_SIMULATIONS = 10000
    
    @classmethod
    def get_current_demand(cls):
        """Calculate current demand from utilization."""
        return cls.U_0 * cls.C_0
    
    @classmethod
    def get_optimized_capacity(cls):
        """Calculate capacity after optimization."""
        return cls.C_0 * (1 + cls.ALPHA_OPT)
    
    @classmethod
    def get_critical_capacity(cls):
        """Calculate critical capacity threshold."""
        return cls.U_c * cls.get_optimized_capacity()

# ============================================================================
# STOCHASTIC MODELS
# ============================================================================

class StochasticGrowthModel:
    """Models uncertainty in traffic growth rates."""
    
    def __init__(self, mean=ModelParameters.GROWTH_MEAN, 
                 std=ModelParameters.GROWTH_STD):
        self.mean = mean
        self.std = std
    
    def sample(self, n_years, n_simulations):
        """
        Generate growth rate samples for multiple years and simulations.
        
        Returns:
            Array of shape (n_simulations, n_years) with growth rates
        """
        # Allow for correlation between years (persistence)
        correlation = 0.3  # Mild positive correlation
        
        # Create correlated growth rates
        growth_rates = np.zeros((n_simulations, n_years))
        
        for sim in range(n_simulations):
            # Generate AR(1) process for growth rates
            growth_rates[sim, 0] = np.random.normal(self.mean, self.std)
            
            for year in range(1, n_years):
                innovation = np.random.normal(0, self.std * np.sqrt(1 - correlation**2))
                growth_rates[sim, year] = (correlation * growth_rates[sim, year-1] + 
                                          (1 - correlation) * self.mean + innovation)
        
        # Ensure growth rates are reasonable (clip extreme values)
        growth_rates = np.clip(growth_rates, -0.02, 0.10)
        
        return growth_rates

class DemandVariabilityModel:
    """Models short-term demand fluctuations."""
    
    def __init__(self, base_demand):
        self.base_demand = base_demand
    
    def sample_hourly(self, hour_of_day, is_weekend=False):
        """
        Sample hourly demand using Poisson distribution.
        
        Args:
            hour_of_day: Hour (0-23)
            is_weekend: Boolean for weekend/weekday
        
        Returns:
            Sampled demand for the hour
        """
        # Determine multiplier based on time
        if is_weekend:
            multiplier = ModelParameters.WEEKEND_MULTIPLIER
        elif 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19:
            multiplier = ModelParameters.PEAK_HOUR_MULTIPLIER
        else:
            multiplier = ModelParameters.OFF_PEAK_MULTIPLIER
        
        # Poisson parameter
        lambda_t = self.base_demand * multiplier / 24
        
        # Sample from Poisson
        return np.random.poisson(lambda_t)
    
    def sample_annual_peak(self, annual_demand):
        """
        Sample peak hour demand for capacity analysis.
        
        Uses Gumbel distribution to model annual maximum.
        """
        # Peak hour is typically 8-10% of daily traffic
        peak_ratio_mean = 0.09
        peak_ratio_std = 0.01
        
        peak_ratio = np.random.normal(peak_ratio_mean, peak_ratio_std)
        peak_ratio = np.clip(peak_ratio, 0.07, 0.12)
        
        # Add extreme value uncertainty
        gumbel_noise = np.random.gumbel(0, 0.05)
        
        return annual_demand * peak_ratio * (1 + gumbel_noise)

class CapacityUncertaintyModel:
    """Models uncertainty in highway capacity."""
    
    def __init__(self, nominal_capacity):
        self.nominal_capacity = nominal_capacity
        self.std = nominal_capacity * ModelParameters.CAPACITY_STD_RATIO
    
    def sample(self, n_samples=1):
        """
        Sample capacity values accounting for various factors.
        
        Factors include:
        - Weather conditions
        - Incident impacts
        - Maintenance activities
        - Seasonal variations
        """
        # Base capacity variation
        base_samples = np.random.normal(self.nominal_capacity, self.std, n_samples)
        
        # Add occasional capacity drops (incidents, weather)
        # 5% chance of 20% capacity reduction
        incident_mask = np.random.random(n_samples) < 0.05
        base_samples[incident_mask] *= 0.8
        
        # Ensure capacity is positive and reasonable
        return np.clip(base_samples, 
                      self.nominal_capacity * 0.7,
                      self.nominal_capacity * 1.1)

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

class MonteCarloSimulation:
    """Main simulation engine for stochastic traffic model."""
    
    def __init__(self, n_simulations=ModelParameters.N_SIMULATIONS):
        self.n_simulations = n_simulations
        self.params = ModelParameters
        
        # Initialize models
        self.growth_model = StochasticGrowthModel()
        self.demand_model = DemandVariabilityModel(self.params.get_current_demand())
        
        # Storage for results
        self.failure_years = []
        self.failure_probabilities = {}
        self.demand_trajectories = []
        self.capacity_trajectories = []
    
    def run_single_simulation(self, sim_id):
        """
        Run a single simulation trajectory.
        
        Returns:
            Year of system failure (or None if no failure in horizon)
        """
        years = self.params.YEARS_TO_SIMULATE
        
        # Sample growth rates for all years
        growth_rates = self.growth_model.sample(years, 1)[0]
        
        # Initialize demand
        current_demand = self.params.get_current_demand()
        demands = [current_demand]
        
        # Determine if optimization happens (80% chance it's implemented)
        implements_optimization = sim_id < self.n_simulations * 0.8
        
        if implements_optimization:
            # Sample implementation delay
            impl_delay = np.random.triangular(2, 3, 5)  # Triangular distribution
            base_capacity = self.params.get_optimized_capacity()
        else:
            impl_delay = 0
            base_capacity = self.params.C_0
        
        capacity_model = CapacityUncertaintyModel(base_capacity)
        
        # Simulate year by year
        for year in range(1, years + 1):
            # Apply growth
            current_demand *= (1 + growth_rates[year-1])
            
            # Don't add extra variability - the demand IS the peak hour demand
            demands.append(current_demand)
            
            # Sample capacity for this year
            if implements_optimization and year <= impl_delay:
                # Before optimization is complete, use base capacity
                capacity = self.params.C_0 * (1 + np.random.normal(0, 0.05))
            else:
                capacity = capacity_model.sample(1)[0]
            
            # Check for failure
            utilization = current_demand / capacity
            if utilization > self.params.U_c:
                return 2025 + year  # Return failure year
        
        # Store trajectory for analysis
        self.demand_trajectories.append(demands)
        
        return None  # No failure in simulation horizon
    
    def run(self):
        """Run full Monte Carlo simulation."""
        print(f"Running {self.n_simulations} Monte Carlo simulations...")
        
        # Progress tracking
        progress_intervals = [int(self.n_simulations * p) for p in [0.25, 0.5, 0.75, 1.0]]
        
        for sim in range(self.n_simulations):
            failure_year = self.run_single_simulation(sim)
            if failure_year:
                self.failure_years.append(failure_year)
            
            # Progress update
            if (sim + 1) in progress_intervals:
                print(f"  Progress: {(sim + 1) / self.n_simulations * 100:.0f}%")
        
        print(f"Completed! {len(self.failure_years)} failures observed.")
        
        # Calculate statistics
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate failure probabilities and statistics."""
        if not self.failure_years:
            print("Warning: No failures observed in simulations!")
            self.stats = {
                'median_failure': None,
                'mean_failure': None,
                'std_failure': None,
                'percentile_5': None,
                'percentile_95': None,
                'p_failure_2030': 0,
                'p_failure_2033': 0,
                'p_failure_2035': 0,
                'total_failures': 0,
                'failure_rate': 0
            }
            return
        
        # Convert to numpy array
        failures = np.array(self.failure_years)
        
        # Calculate cumulative probabilities
        years = range(2025, 2045)
        for year in years:
            n_failed = np.sum(failures <= year)
            self.failure_probabilities[year] = n_failed / self.n_simulations
        
        # Key statistics
        self.stats = {
            'median_failure': np.median(failures) if len(failures) > 0 else None,
            'mean_failure': np.mean(failures) if len(failures) > 0 else None,
            'std_failure': np.std(failures) if len(failures) > 0 else None,
            'percentile_5': np.percentile(failures, 5) if len(failures) > 0 else None,
            'percentile_95': np.percentile(failures, 95) if len(failures) > 0 else None,
            'p_failure_2030': self.failure_probabilities.get(2030, 0),
            'p_failure_2033': self.failure_probabilities.get(2033, 0),
            'p_failure_2035': self.failure_probabilities.get(2035, 0),
            'total_failures': len(failures),
            'failure_rate': len(failures) / self.n_simulations
        }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_failure_distribution_plot(simulation):
    """Create histogram of failure year distribution."""
    print("Generating failure distribution plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Histogram of failure years
    if simulation.failure_years:
        failures = np.array(simulation.failure_years)
        
        # Create histogram
        counts, bins, patches = ax1.hist(failures, bins=20, density=True, 
                                         alpha=0.7, color='darkred', edgecolor='black')
        
        # Fit and plot normal distribution
        mu, sigma = stats.norm.fit(failures)
        x = np.linspace(failures.min(), failures.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Normal fit: μ={mu:.1f}, σ={sigma:.1f}')
        
        # Add vertical lines for key percentiles
        for p, label, color in [(5, '5th %ile', 'blue'), 
                                (50, 'Median', 'black'), 
                                (95, '95th %ile', 'green')]:
            val = np.percentile(failures, p)
            ax1.axvline(val, color=color, linestyle='--', alpha=0.7, label=f'{label}: {val:.1f}')
        
        # Formatting
        ax1.set_xlabel('Year of System Failure', fontsize=11)
        ax1.set_ylabel('Probability Density', fontsize=11)
        ax1.set_title('Distribution of System Failure Times', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Add text box with key stats
        stats_text = f'Mean: {mu:.1f}\nMedian: {np.median(failures):.1f}\n'
        stats_text += f'Std Dev: {sigma:.1f}\n'
        stats_text += f'95% CI: [{simulation.stats["percentile_5"]:.1f}, {simulation.stats["percentile_95"]:.1f}]'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    # Subplot 2: Cumulative failure probability
    years = sorted(simulation.failure_probabilities.keys())
    probs = [simulation.failure_probabilities[y] for y in years]
    
    ax2.plot(years, probs, 'b-', linewidth=3, marker='o', markersize=5, markevery=2)
    
    # Add shaded regions
    ax2.fill_between(years, 0, probs, alpha=0.3, color='lightblue')
    
    # Mark key probability thresholds
    for prob, label, color in [(0.5, '50% probability', 'orange'),
                               (0.8, '80% probability', 'red')]:
        ax2.axhline(prob, color=color, linestyle=':', alpha=0.7, label=label)
        
        # Find year where this probability is reached
        for i, p in enumerate(probs):
            if p >= prob:
                ax2.plot(years[i], prob, 'o', color=color, markersize=8)
                ax2.annotate(f'{years[i]}', xy=(years[i], prob),
                           xytext=(years[i]-1, prob+0.05),
                           arrowprops=dict(arrowstyle='->', color=color),
                           fontsize=9)
                break
    
    # Mark specific years
    for year in [2030, 2033, 2035]:
        if year in simulation.failure_probabilities:
            prob = simulation.failure_probabilities[year]
            ax2.plot(year, prob, 'ko', markersize=6)
            ax2.text(year, prob-0.05, f'{prob:.1%}', ha='center', fontsize=8)
    
    # Formatting
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Cumulative Failure Probability', fontsize=11)
    ax2.set_title('Cumulative Probability of System Failure', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(2024, 2041)
    
    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.suptitle('Stochastic Analysis of Highway System Failure', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = FIGURES_DIR / 'fig_15_failure_probability_distribution.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def create_sensitivity_analysis_plot(base_simulation):
    """Create sensitivity analysis showing impact of parameter variations."""
    print("Generating sensitivity analysis plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Define parameter variations to test
    variations = {
        'Growth Rate': {
            'parameter': 'growth_mean',
            'values': [0.025, 0.030, 0.035, 0.040, 0.045],
            'labels': ['2.5%', '3.0%', '3.5%', '4.0%', '4.5%'],
            'ax': axes[0, 0]
        },
        'Optimization Gain': {
            'parameter': 'alpha_opt',
            'values': [0.25, 0.30, 0.35, 0.40, 0.45],
            'labels': ['25%', '30%', '35%', '40%', '45%'],
            'ax': axes[0, 1]
        },
        'Critical Utilization': {
            'parameter': 'u_critical',
            'values': [0.90, 0.925, 0.95, 0.975, 0.99],
            'labels': ['90%', '92.5%', '95%', '97.5%', '99%'],
            'ax': axes[1, 0]
        },
        'Growth Volatility': {
            'parameter': 'growth_std',
            'values': [0.005, 0.0075, 0.010, 0.0125, 0.015],
            'labels': ['0.5%', '0.75%', '1.0%', '1.25%', '1.5%'],
            'ax': axes[1, 1]
        }
    }
    
    # Run sensitivity for each parameter
    for param_name, config in variations.items():
        ax = config['ax']
        median_failures = []
        p_2033_failures = []
        
        for value in config['values']:
            # Create modified simulation
            sim = MonteCarloSimulation(n_simulations=1000)  # Fewer for speed
            
            # Modify parameter
            if config['parameter'] == 'growth_mean':
                sim.growth_model.mean = value
            elif config['parameter'] == 'alpha_opt':
                ModelParameters.ALPHA_OPT = value
            elif config['parameter'] == 'u_critical':
                ModelParameters.U_c = value
            elif config['parameter'] == 'growth_std':
                sim.growth_model.std = value
            
            # Run simulation
            sim.run()
            
            # Collect results
            if sim.failure_years:
                median_failures.append(np.median(sim.failure_years))
                p_2033_failures.append(sim.failure_probabilities.get(2033, 0))
            else:
                median_failures.append(2045)  # No failure
                p_2033_failures.append(0)
        
        # Reset parameter
        if config['parameter'] == 'alpha_opt':
            ModelParameters.ALPHA_OPT = 0.35
        elif config['parameter'] == 'u_critical':
            ModelParameters.U_c = 0.95
        
        # Plot results
        x = range(len(config['values']))
        
        # Create twin axis
        ax2 = ax.twinx()
        
        # Plot median failure year
        line1 = ax.plot(x, median_failures, 'b-o', linewidth=2, markersize=8, 
                       label='Median Failure Year')
        ax.set_ylabel('Median Failure Year', color='b', fontsize=10)
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot P(failure by 2033)
        line2 = ax2.plot(x, p_2033_failures, 'r-s', linewidth=2, markersize=8,
                        label='P(Failure by 2033)')
        ax2.set_ylabel('P(Failure by 2033)', color='r', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Formatting
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_title(f'Sensitivity to {param_name}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(config['labels'])
        ax.grid(True, alpha=0.3)
        
        # Mark baseline
        baseline_idx = 2  # Middle value is baseline
        ax.axvline(baseline_idx, color='gray', linestyle=':', alpha=0.5)
        ax.text(baseline_idx, ax.get_ylim()[0], 'Baseline', ha='center', 
               fontsize=8, color='gray')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize=8)
    
    plt.suptitle('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = FIGURES_DIR / 'fig_16_sensitivity_analysis.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("STOCHASTIC TRAFFIC MODEL - MONTE CARLO SIMULATION")
    print("Task 4.2: Uncertainty Quantification")
    print("="*60 + "\n")
    
    # Run main simulation
    print("Initializing simulation parameters...")
    simulation = MonteCarloSimulation()
    
    print(f"Parameters:")
    print(f"  Current utilization: {ModelParameters.U_0:.1%}")
    print(f"  Growth rate: {ModelParameters.GROWTH_MEAN:.1%} ± {ModelParameters.GROWTH_STD:.1%}")
    print(f"  Optimization gain: {ModelParameters.ALPHA_OPT:.1%}")
    print(f"  Critical threshold: {ModelParameters.U_c:.1%}")
    print(f"  Simulations: {ModelParameters.N_SIMULATIONS:,}")
    print()
    
    # Run simulation
    simulation.run()
    
    # Display results
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
    if simulation.stats['median_failure']:
        print(f"\nFailure Time Statistics:")
        print(f"  Median failure year: {simulation.stats['median_failure']:.1f}")
        print(f"  Mean failure year: {simulation.stats['mean_failure']:.1f}")
        print(f"  Standard deviation: {simulation.stats['std_failure']:.1f} years")
        print(f"  5th percentile: {simulation.stats['percentile_5']:.1f}")
        print(f"  95th percentile: {simulation.stats['percentile_95']:.1f}")
        
        print(f"\nCumulative Failure Probabilities:")
        print(f"  P(failure by 2030): {simulation.stats['p_failure_2030']:.1%}")
        print(f"  P(failure by 2033): {simulation.stats['p_failure_2033']:.1%}")
        print(f"  P(failure by 2035): {simulation.stats['p_failure_2035']:.1%}")
        
        print(f"\nOverall Statistics:")
        print(f"  Total failures: {simulation.stats['total_failures']:,} / {simulation.n_simulations:,}")
        print(f"  Failure rate: {simulation.stats['failure_rate']:.1%}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_failure_distribution_plot(simulation)
    create_sensitivity_analysis_plot(simulation)
    
    # Save results to JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_simulations': ModelParameters.N_SIMULATIONS,
            'current_utilization': ModelParameters.U_0,
            'growth_mean': ModelParameters.GROWTH_MEAN,
            'growth_std': ModelParameters.GROWTH_STD,
            'optimization_gain': ModelParameters.ALPHA_OPT,
            'critical_utilization': ModelParameters.U_c,
            'implementation_delay': ModelParameters.IMPLEMENTATION_DELAY
        },
        'results': {
            'median_failure_year': float(simulation.stats['median_failure']) if simulation.stats['median_failure'] else None,
            'mean_failure_year': float(simulation.stats['mean_failure']) if simulation.stats['mean_failure'] else None,
            'std_failure_year': float(simulation.stats['std_failure']) if simulation.stats['std_failure'] else None,
            'percentile_5': float(simulation.stats['percentile_5']) if simulation.stats['percentile_5'] else None,
            'percentile_95': float(simulation.stats['percentile_95']) if simulation.stats['percentile_95'] else None,
            'p_failure_2030': simulation.stats['p_failure_2030'],
            'p_failure_2033': simulation.stats['p_failure_2033'],
            'p_failure_2035': simulation.stats['p_failure_2035'],
            'total_failures': simulation.stats['total_failures'],
            'failure_rate': simulation.stats['failure_rate']
        },
        'failure_probabilities_by_year': {
            str(year): prob for year, prob in simulation.failure_probabilities.items()
        }
    }
    
    output_file = REPORTS_DIR / 'stochastic_model_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*60)
    print("STOCHASTIC MODEL ANALYSIS COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()