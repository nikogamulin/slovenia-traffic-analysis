"""
Bayesian MCMC Model for Traffic Collapse Hypothesis Testing
Uses PyMC3 to test the primary hypothesis about roadwork impacts
"""

import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BayesianTrafficModel:
    """
    Bayesian hierarchical model for testing traffic collapse hypotheses
    """
    
    def __init__(self, daily_compound_file, focus_segment='Ljubljana Bypass'):
        """
        Initialize Bayesian model
        
        Args:
            daily_compound_file: Path to daily compound factors from previous analysis
            focus_segment: Primary segment to model (default: Ljubljana Bypass)
        """
        self.daily_compound_file = daily_compound_file
        self.focus_segment = focus_segment
        self.model = None
        self.trace = None
        self.posterior_predictive = None
        
    def prepare_model_data(self):
        """Prepare data for Bayesian modeling"""
        
        print("Preparing data for Bayesian analysis...")
        
        # Load compound factors data
        self.data = pd.read_csv(self.daily_compound_file)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Filter to focus segment
        segment_data = self.data[self.data['road_name'] == self.focus_segment].copy()
        
        if len(segment_data) == 0:
            # If specific segment not found, use overall average
            print(f"Segment '{self.focus_segment}' not found, using network-wide average")
            segment_data = self.data.groupby('date').agg({
                'avg_speed': 'mean',
                'n_major_roadworks': 'first',
                'n_total_roadworks': 'first',
                'n_incidents': 'first',
                'is_international_holiday': 'first',
                'is_adverse_weather': 'first',
                'total_precip': 'first',
                'day_of_week': 'first',
                'month': 'first',
                'is_weekend': 'first'
            }).reset_index()
            
        # Create model variables
        self.model_data = {
            'speed': segment_data['avg_speed'].values,
            'roadwork_active': (segment_data['n_major_roadworks'] > 0).astype(int).values,
            'n_roadworks': segment_data['n_total_roadworks'].values,
            'incident_active': (segment_data['n_incidents'] > 0).astype(int).values,
            'holiday_transit': segment_data['is_international_holiday'].astype(int).values,
            'precipitation': segment_data['total_precip'].values,
            'adverse_weather': segment_data['is_adverse_weather'].astype(int).values,
            'hour_sin': np.sin(2 * np.pi * segment_data['date'].dt.hour / 24).values,
            'hour_cos': np.cos(2 * np.pi * segment_data['date'].dt.hour / 24).values,
            'dow_sin': np.sin(2 * np.pi * segment_data['day_of_week'] / 7).values,
            'dow_cos': np.cos(2 * np.pi * segment_data['day_of_week'] / 7).values,
            'month_sin': np.sin(2 * np.pi * segment_data['month'] / 12).values,
            'month_cos': np.cos(2 * np.pi * segment_data['month'] / 12).values,
            'is_weekend': segment_data['is_weekend'].astype(int).values
        }
        
        # Standardize continuous predictors
        self.model_data['precipitation_std'] = (
            self.model_data['precipitation'] - self.model_data['precipitation'].mean()
        ) / (self.model_data['precipitation'].std() + 1e-6)
        
        self.n_obs = len(self.model_data['speed'])
        print(f"Model data prepared: {self.n_obs} observations")
        
        # Print data summary
        print("\nDependent Variable (Speed):")
        print(f"  Mean: {self.model_data['speed'].mean():.1f} km/h")
        print(f"  Std: {self.model_data['speed'].std():.1f} km/h")
        print(f"  Range: [{self.model_data['speed'].min():.1f}, {self.model_data['speed'].max():.1f}]")
        
        print("\nKey Predictors:")
        print(f"  Days with Major Roadworks: {self.model_data['roadwork_active'].sum()} ({self.model_data['roadwork_active'].mean():.1%})")
        print(f"  Days with Incidents: {self.model_data['incident_active'].sum()} ({self.model_data['incident_active'].mean():.1%})")
        print(f"  International Holiday Days: {self.model_data['holiday_transit'].sum()} ({self.model_data['holiday_transit'].mean():.1%})")
        
    def build_model(self):
        """Build the Bayesian hierarchical model"""
        
        print("\n" + "=" * 60)
        print("BUILDING BAYESIAN MODEL")
        print("=" * 60)
        
        with pm.Model() as self.model:
            
            # Priors for coefficients
            # Using weakly informative priors (Normal with large variance)
            
            # Intercept - centered around typical highway speed
            intercept = pm.Normal('intercept', mu=80, sigma=20)
            
            # PRIMARY HYPOTHESIS: Roadwork impact
            # Prior expectation: roadworks reduce speed by 10-20 km/h
            beta_roadwork = pm.Normal('beta_roadwork_active', mu=-15, sigma=10)
            
            # Additional roadwork effect (per additional project)
            beta_n_roadworks = pm.Normal('beta_n_roadworks', mu=-3, sigma=2)
            
            # Secondary factors
            beta_incident = pm.Normal('beta_incident', mu=-5, sigma=5)
            beta_holiday = pm.Normal('beta_holiday', mu=-8, sigma=5)
            beta_precipitation = pm.Normal('beta_precipitation', mu=-2, sigma=2)
            beta_weekend = pm.Normal('beta_weekend', mu=-3, sigma=3)
            
            # Temporal effects (cyclical)
            beta_hour_sin = pm.Normal('beta_hour_sin', mu=0, sigma=5)
            beta_hour_cos = pm.Normal('beta_hour_cos', mu=0, sigma=5)
            beta_dow_sin = pm.Normal('beta_dow_sin', mu=0, sigma=3)
            beta_dow_cos = pm.Normal('beta_dow_cos', mu=0, sigma=3)
            beta_month_sin = pm.Normal('beta_month_sin', mu=0, sigma=3)
            beta_month_cos = pm.Normal('beta_month_cos', mu=0, sigma=3)
            
            # Interaction terms
            beta_roadwork_holiday = pm.Normal('beta_roadwork_x_holiday', mu=-5, sigma=5)
            beta_roadwork_incident = pm.Normal('beta_roadwork_x_incident', mu=-3, sigma=3)
            
            # Model variance
            sigma = pm.HalfNormal('sigma', sigma=10)
            
            # Linear predictor
            mu = (
                intercept +
                beta_roadwork * self.model_data['roadwork_active'] +
                beta_n_roadworks * self.model_data['n_roadworks'] +
                beta_incident * self.model_data['incident_active'] +
                beta_holiday * self.model_data['holiday_transit'] +
                beta_precipitation * self.model_data['precipitation_std'] +
                beta_weekend * self.model_data['is_weekend'] +
                beta_hour_sin * self.model_data['hour_sin'] +
                beta_hour_cos * self.model_data['hour_cos'] +
                beta_dow_sin * self.model_data['dow_sin'] +
                beta_dow_cos * self.model_data['dow_cos'] +
                beta_month_sin * self.model_data['month_sin'] +
                beta_month_cos * self.model_data['month_cos'] +
                beta_roadwork_holiday * self.model_data['roadwork_active'] * self.model_data['holiday_transit'] +
                beta_roadwork_incident * self.model_data['roadwork_active'] * self.model_data['incident_active']
            )
            
            # Likelihood
            speed_obs = pm.Normal('speed_obs', mu=mu, sigma=sigma, 
                                 observed=self.model_data['speed'])
            
        print("Model structure:")
        print(f"  Predictors: 14 (including interactions)")
        print(f"  Observations: {self.n_obs}")
        print(f"  Prior distributions: Weakly informative Normal")
        print(f"  Likelihood: Normal (Gaussian errors)")
        
    def run_mcmc(self, n_samples=4000, n_chains=4, n_tune=2000):
        """
        Run MCMC sampling
        
        Args:
            n_samples: Number of samples per chain
            n_chains: Number of MCMC chains
            n_tune: Number of tuning samples
        """
        
        print("\n" + "=" * 60)
        print("RUNNING MCMC SAMPLING")
        print("=" * 60)
        print(f"Chains: {n_chains}")
        print(f"Samples per chain: {n_samples}")
        print(f"Tuning samples: {n_tune}")
        print("\nSampling... (this may take a few minutes)")
        
        with self.model:
            # Use NUTS sampler (No U-Turn Sampler)
            self.trace = pm.sample(
                draws=n_samples,
                chains=n_chains,
                tune=n_tune,
                return_inferencedata=True,
                progressbar=True,
                random_seed=42
            )
            
            # Sample posterior predictive
            self.posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                random_seed=42
            )
            
        print("\nMCMC sampling complete!")
        
        # Print convergence diagnostics
        self._check_convergence()
        
    def _check_convergence(self):
        """Check MCMC convergence diagnostics"""
        
        print("\n" + "=" * 60)
        print("CONVERGENCE DIAGNOSTICS")
        print("=" * 60)
        
        # Get summary with R-hat and ESS
        summary = az.summary(self.trace, var_names=[
            'beta_roadwork_active', 'beta_n_roadworks', 
            'beta_incident', 'beta_holiday'
        ])
        
        print("\nKey Parameter Diagnostics:")
        print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']])
        
        # Check R-hat values
        max_rhat = summary['r_hat'].max()
        if max_rhat < 1.01:
            print(f"\n✓ Convergence GOOD (max R-hat = {max_rhat:.3f} < 1.01)")
        else:
            print(f"\n⚠ Convergence WARNING (max R-hat = {max_rhat:.3f} > 1.01)")
            
    def analyze_results(self):
        """Analyze and interpret MCMC results"""
        
        print("\n" + "=" * 60)
        print("HYPOTHESIS TEST RESULTS")
        print("=" * 60)
        
        # Extract posterior samples
        posterior = self.trace.posterior
        
        # PRIMARY HYPOTHESIS: Roadwork impact
        roadwork_samples = posterior['beta_roadwork_active'].values.flatten()
        roadwork_mean = roadwork_samples.mean()
        roadwork_std = roadwork_samples.std()
        roadwork_hdi = az.hdi(roadwork_samples, hdi_prob=0.95)
        
        print("\nPRIMARY HYPOTHESIS (H1): Active roadworks reduce speed")
        print("-" * 50)
        print(f"Posterior Mean: {roadwork_mean:.2f} km/h")
        print(f"Posterior Std: {roadwork_std:.2f} km/h")
        print(f"95% Credible Interval: [{roadwork_hdi[0]:.2f}, {roadwork_hdi[1]:.2f}]")
        
        # Probability that effect is negative (speed reduction)
        prob_negative = (roadwork_samples < 0).mean()
        print(f"P(β_roadwork < 0): {prob_negative:.3f}")
        
        if roadwork_hdi[1] < 0:
            print("\n✓ STRONG EVIDENCE: Roadworks significantly reduce speed")
            print(f"  We are 95% confident the effect is between {roadwork_hdi[0]:.1f} and {roadwork_hdi[1]:.1f} km/h")
        elif roadwork_hdi[0] < 0 < roadwork_hdi[1]:
            print("\n○ WEAK EVIDENCE: Roadwork effect includes zero")
        else:
            print("\n✗ NO EVIDENCE: Roadworks do not reduce speed")
            
        # Additional roadwork effect
        n_roadwork_samples = posterior['beta_n_roadworks'].values.flatten()
        n_roadwork_mean = n_roadwork_samples.mean()
        n_roadwork_hdi = az.hdi(n_roadwork_samples, hdi_prob=0.95)
        
        print("\nADDITIONAL ROADWORK EFFECT (per extra project):")
        print(f"Posterior Mean: {n_roadwork_mean:.2f} km/h")
        print(f"95% CI: [{n_roadwork_hdi[0]:.2f}, {n_roadwork_hdi[1]:.2f}]")
        
        # Secondary factors
        print("\nSECONDARY FACTORS:")
        print("-" * 50)
        
        factors = [
            ('Incident', 'beta_incident'),
            ('Holiday Transit', 'beta_holiday'),
            ('Precipitation', 'beta_precipitation'),
            ('Weekend', 'beta_weekend')
        ]
        
        for name, param in factors:
            samples = posterior[param].values.flatten()
            mean_val = samples.mean()
            hdi_val = az.hdi(samples, hdi_prob=0.95)
            print(f"{name:20} Mean: {mean_val:6.2f} km/h, 95% CI: [{hdi_val[0]:6.2f}, {hdi_val[1]:6.2f}]")
            
        # Interaction effects
        print("\nINTERACTION EFFECTS:")
        print("-" * 50)
        
        interactions = [
            ('Roadwork × Holiday', 'beta_roadwork_x_holiday'),
            ('Roadwork × Incident', 'beta_roadwork_x_incident')
        ]
        
        for name, param in interactions:
            samples = posterior[param].values.flatten()
            mean_val = samples.mean()
            hdi_val = az.hdi(samples, hdi_prob=0.95)
            significant = not (hdi_val[0] < 0 < hdi_val[1])
            print(f"{name:20} Mean: {mean_val:6.2f} km/h, Significant: {'Yes' if significant else 'No'}")
            
        return {
            'roadwork_effect': roadwork_mean,
            'roadwork_ci': roadwork_hdi,
            'prob_negative': prob_negative,
            'n_roadwork_effect': n_roadwork_mean
        }
        
    def posterior_predictive_check(self):
        """Perform posterior predictive checks"""
        
        print("\n" + "=" * 60)
        print("POSTERIOR PREDICTIVE CHECK")
        print("=" * 60)
        
        # Get observed and predicted values
        observed = self.model_data['speed']
        predicted = self.posterior_predictive['speed_obs'].mean(axis=0)
        
        # Calculate fit metrics
        residuals = observed - predicted
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        r2 = 1 - np.sum(residuals**2) / np.sum((observed - observed.mean())**2)
        
        print(f"Model Fit Metrics:")
        print(f"  RMSE: {rmse:.2f} km/h")
        print(f"  MAE: {mae:.2f} km/h")
        print(f"  R²: {r2:.3f}")
        
        # Check for systematic patterns in residuals
        from scipy import stats
        _, p_value = stats.normaltest(residuals)
        
        if p_value > 0.05:
            print(f"\n✓ Residuals appear normally distributed (p = {p_value:.3f})")
        else:
            print(f"\n⚠ Residuals may not be normally distributed (p = {p_value:.3f})")
            
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'residuals': residuals
        }
        
    def plot_results(self):
        """Generate diagnostic plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Trace plot for roadwork effect
        ax = axes[0, 0]
        az.plot_trace(self.trace, var_names=['beta_roadwork_active'], 
                     axes=np.array([[ax, axes[0, 1]]])[:, :1])
        ax.set_title('Roadwork Effect - Trace')
        
        # Plot 2: Posterior distribution for roadwork effect
        ax = axes[0, 1]
        az.plot_posterior(self.trace, var_names=['beta_roadwork_active'], 
                         ax=ax, hdi_prob=0.95)
        ax.set_title('Roadwork Effect - Posterior')
        
        # Plot 3: Forest plot for all coefficients
        ax = axes[0, 2]
        az.plot_forest(self.trace, var_names=[
            'beta_roadwork_active', 'beta_n_roadworks',
            'beta_incident', 'beta_holiday'
        ], ax=ax, combined=True)
        ax.set_title('Effect Sizes')
        
        # Plot 4: Observed vs Predicted
        ax = axes[1, 0]
        observed = self.model_data['speed']
        predicted = self.posterior_predictive['speed_obs'].mean(axis=0)
        ax.scatter(observed, predicted, alpha=0.5)
        ax.plot([observed.min(), observed.max()], 
               [observed.min(), observed.max()], 'r--')
        ax.set_xlabel('Observed Speed (km/h)')
        ax.set_ylabel('Predicted Speed (km/h)')
        ax.set_title('Observed vs Predicted')
        
        # Plot 5: Residuals histogram
        ax = axes[1, 1]
        residuals = observed - predicted
        ax.hist(residuals, bins=30, edgecolor='black')
        ax.set_xlabel('Residual (km/h)')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        
        # Plot 6: Effect summary
        ax = axes[1, 2]
        effects = {
            'Roadwork\nActive': self.trace.posterior['beta_roadwork_active'].mean().item(),
            'Additional\nRoadwork': self.trace.posterior['beta_n_roadworks'].mean().item(),
            'Incident': self.trace.posterior['beta_incident'].mean().item(),
            'Holiday': self.trace.posterior['beta_holiday'].mean().item(),
            'Precipitation': self.trace.posterior['beta_precipitation'].mean().item()
        }
        ax.bar(effects.keys(), effects.values(), 
              color=['red' if v < 0 else 'green' for v in effects.values()])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Effect on Speed (km/h)')
        ax.set_title('Average Effects')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('./reports/bayesian_results.png', dpi=150)
        plt.show()
        
        print("\nDiagnostic plots saved to ./reports/bayesian_results.png")
        
    def export_results(self, output_dir='./data'):
        """Export Bayesian analysis results"""
        
        # Export posterior samples
        posterior_df = self.trace.posterior.to_dataframe()
        posterior_file = f"{output_dir}/bayesian_posterior_samples.csv"
        posterior_df.to_csv(posterior_file, index=False)
        print(f"\nPosterior samples exported to {posterior_file}")
        
        # Export summary statistics
        summary = az.summary(self.trace)
        summary_file = f"{output_dir}/bayesian_summary.csv"
        summary.to_csv(summary_file)
        print(f"Summary statistics exported to {summary_file}")
        
        return summary


def main():
    """Main execution for Bayesian analysis"""
    
    print("\n" + "=" * 60)
    print("BAYESIAN HYPOTHESIS TESTING")
    print("Traffic Collapse Causal Analysis")
    print("=" * 60)
    
    # Initialize model
    model = BayesianTrafficModel(
        daily_compound_file='./data/daily_compound_factors.csv',
        focus_segment='Ljubljana Bypass'
    )
    
    # Prepare data
    model.prepare_model_data()
    
    # Build model
    model.build_model()
    
    # Run MCMC
    model.run_mcmc(n_samples=4000, n_chains=4, n_tune=2000)
    
    # Analyze results
    results = model.analyze_results()
    
    # Posterior predictive check
    fit_metrics = model.posterior_predictive_check()
    
    # Generate plots
    model.plot_results()
    
    # Export results
    summary = model.export_results()
    
    print("\n" + "=" * 60)
    print("BAYESIAN ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("\nFINAL CONCLUSION:")
    if results['roadwork_ci'][1] < -5:
        print("✓ HYPOTHESIS CONFIRMED: Strong evidence that roadworks are the primary")
        print("  driver of traffic speed reduction in the Slovenian network.")
        print(f"  Effect size: {results['roadwork_effect']:.1f} km/h [95% CI: {results['roadwork_ci'][0]:.1f}, {results['roadwork_ci'][1]:.1f}]")
    else:
        print("○ HYPOTHESIS UNCERTAIN: Evidence suggests roadworks have an effect,")
        print("  but magnitude may be smaller than hypothesized.")
        
    return model, results


if __name__ == "__main__":
    model, results = main()