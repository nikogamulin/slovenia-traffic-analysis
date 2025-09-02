#!/usr/bin/env python3
"""
Network Resilience Model - Cascade Effects Quantification
Task 4.3: Model incident cascade propagation and network vulnerability

This script implements a graph-based network model to quantify cascade effects
in Slovenia's highway network. It calculates betweenness centrality, models
incident propagation, and validates against the observed 33% bidirectional 
impact rate and 11.1% cascade rate.

Key components:
- Highway network graph construction
- Betweenness centrality analysis
- Cascade propagation simulation
- Resilience metrics calculation
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial import distance
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
# NETWORK TOPOLOGY DEFINITION
# ============================================================================

class SloveniaHighwayNetwork:
    """Define Slovenia's highway network topology."""
    
    def __init__(self):
        """Initialize network with nodes and edges based on real highway system."""
        self.G = nx.DiGraph()
        self._build_network()
        
    def _build_network(self):
        """Build the highway network graph."""
        
        # Major nodes (cities/junctions)
        nodes = {
            # A1 Highway (E57, E70)
            'Šentilj': {'pos': (15.65, 46.68), 'type': 'border', 'country': 'AT'},
            'Maribor': {'pos': (15.65, 46.55), 'type': 'city'},
            'Celje': {'pos': (15.26, 46.23), 'type': 'city'},
            'Ljubljana': {'pos': (14.51, 46.05), 'type': 'capital'},
            'Postojna': {'pos': (14.21, 45.77), 'type': 'junction'},
            'Koper': {'pos': (13.73, 45.55), 'type': 'port'},
            
            # A2 Highway (E61, E70)
            'Karavanke': {'pos': (14.00, 46.44), 'type': 'border', 'country': 'AT'},
            'Kranj': {'pos': (14.36, 46.24), 'type': 'city'},
            'Ljubljana_Ring': {'pos': (14.51, 46.08), 'type': 'junction'},
            'Novo_Mesto': {'pos': (15.17, 45.80), 'type': 'city'},
            'Obrežje': {'pos': (15.69, 45.85), 'type': 'border', 'country': 'HR'},
            
            # A4 Highway
            'Gruškovje': {'pos': (15.89, 46.26), 'type': 'border', 'country': 'HR'},
            'Ptuj': {'pos': (15.87, 46.42), 'type': 'city'},
            
            # A5 Highway
            'Dragotinci': {'pos': (16.00, 46.50), 'type': 'border', 'country': 'HU'},
            'Murska_Sobota': {'pos': (16.17, 46.66), 'type': 'city'},
            
            # H4 Express Road
            'Sežana': {'pos': (13.87, 45.71), 'type': 'junction'},
            'Fernetiči': {'pos': (13.73, 45.59), 'type': 'border', 'country': 'IT'},
        }
        
        # Add nodes to graph
        for node, attrs in nodes.items():
            self.G.add_node(node, **attrs)
        
        # Define edges with distances (km) and capacities (veh/hr)
        edges = [
            # A1 main corridor
            ('Šentilj', 'Maribor', {'distance': 15, 'capacity': 6600, 'highway': 'A1'}),
            ('Maribor', 'Celje', {'distance': 65, 'capacity': 6600, 'highway': 'A1'}),
            ('Celje', 'Ljubljana', {'distance': 75, 'capacity': 9900, 'highway': 'A1'}),
            ('Ljubljana', 'Postojna', {'distance': 50, 'capacity': 9900, 'highway': 'A1'}),
            ('Postojna', 'Koper', {'distance': 65, 'capacity': 6600, 'highway': 'A1'}),
            
            # A2 main corridor
            ('Karavanke', 'Kranj', {'distance': 35, 'capacity': 6600, 'highway': 'A2'}),
            ('Kranj', 'Ljubljana_Ring', {'distance': 20, 'capacity': 9900, 'highway': 'A2'}),
            ('Ljubljana_Ring', 'Novo_Mesto', {'distance': 70, 'capacity': 6600, 'highway': 'A2'}),
            ('Novo_Mesto', 'Obrežje', {'distance': 45, 'capacity': 6600, 'highway': 'A2'}),
            
            # Cross connections
            ('Ljubljana', 'Ljubljana_Ring', {'distance': 5, 'capacity': 13200, 'highway': 'Ring'}),
            ('Postojna', 'Sežana', {'distance': 25, 'capacity': 4400, 'highway': 'H4'}),
            ('Sežana', 'Fernetiči', {'distance': 15, 'capacity': 4400, 'highway': 'H4'}),
            
            # A4 connections
            ('Maribor', 'Ptuj', {'distance': 25, 'capacity': 4400, 'highway': 'A4'}),
            ('Ptuj', 'Gruškovje', {'distance': 35, 'capacity': 4400, 'highway': 'A4'}),
            
            # A5 connections  
            ('Maribor', 'Murska_Sobota', {'distance': 60, 'capacity': 4400, 'highway': 'A5'}),
            ('Murska_Sobota', 'Dragotinci', {'distance': 35, 'capacity': 4400, 'highway': 'A5'}),
        ]
        
        # Add bidirectional edges
        for u, v, attrs in edges:
            self.G.add_edge(u, v, **attrs)
            self.G.add_edge(v, u, **attrs)
        
    def get_graph(self):
        """Return the network graph."""
        return self.G

# ============================================================================
# CENTRALITY AND VULNERABILITY ANALYSIS
# ============================================================================

class NetworkVulnerabilityAnalyzer:
    """Analyze network vulnerability using graph metrics."""
    
    def __init__(self, graph):
        self.G = graph
        self.centrality_metrics = {}
        
    def calculate_betweenness_centrality(self, weight='distance'):
        """Calculate betweenness centrality for nodes and edges."""
        # Node betweenness
        self.node_betweenness = nx.betweenness_centrality(
            self.G, weight=weight, normalized=True
        )
        
        # Edge betweenness
        self.edge_betweenness = nx.edge_betweenness_centrality(
            self.G, weight=weight, normalized=True
        )
        
        return self.node_betweenness, self.edge_betweenness
    
    def calculate_vulnerability_index(self):
        """Calculate vulnerability index for each segment."""
        vulnerability = {}
        
        for edge in self.G.edges():
            # Factors contributing to vulnerability
            betweenness = self.edge_betweenness.get(edge, 0)
            capacity = self.G[edge[0]][edge[1]]['capacity']
            
            # Alternative routes availability
            G_temp = self.G.copy()
            G_temp.remove_edge(edge[0], edge[1])
            try:
                # Check if alternative path exists
                alt_path = nx.shortest_path(G_temp, edge[0], edge[1])
                alt_factor = 1.0 / len(alt_path) if len(alt_path) > 0 else 0
            except nx.NetworkXNoPath:
                alt_factor = 1.0  # No alternative = high vulnerability
            
            # Calculate vulnerability score
            vulnerability[edge] = {
                'betweenness': betweenness,
                'capacity_factor': 1.0 / (capacity / 6600),  # Normalized by standard capacity
                'alternative_factor': alt_factor,
                'total_score': betweenness * (1.0 / (capacity / 6600)) * (1 + alt_factor)
            }
        
        return vulnerability
    
    def identify_critical_segments(self, top_n=5):
        """Identify the most critical segments."""
        vulnerability = self.calculate_vulnerability_index()
        
        # Sort by total vulnerability score
        sorted_segments = sorted(
            vulnerability.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        return sorted_segments[:top_n]

# ============================================================================
# CASCADE PROPAGATION MODEL
# ============================================================================

class CascadePropagationModel:
    """Model cascade effects from incidents."""
    
    def __init__(self, graph):
        self.G = graph
        
        # Calibrated parameters from observed data
        self.BIDIRECTIONAL_IMPACT_RATE = 0.33  # 33% affect both directions
        self.CASCADE_RATE = 0.111  # 11.1% cascade within 2 hours
        self.AVG_CLEARANCE_TIME = 43.4  # minutes
        self.RUBBERNECKING_FACTOR = 0.15  # 15% capacity reduction in opposite direction
        
    def calculate_cascade_probability(self, location, time_of_day, severity):
        """
        Calculate probability of cascade given incident parameters.
        
        Args:
            location: Edge in the graph
            time_of_day: Hour (0-23)
            severity: 'minor', 'major', 'fatal'
        
        Returns:
            Probability of cascade
        """
        # Base probabilities by severity
        severity_prob = {
            'minor': 0.05,
            'major': 0.15,
            'fatal': 0.30
        }
        
        # Time factor (peak hours have higher cascade probability)
        if 7 <= time_of_day <= 9 or 17 <= time_of_day <= 19:
            time_factor = 1.5
        elif 10 <= time_of_day <= 16:
            time_factor = 1.2
        else:
            time_factor = 0.8
        
        # Location factor (high centrality = higher cascade risk)
        analyzer = NetworkVulnerabilityAnalyzer(self.G)
        _, edge_betweenness = analyzer.calculate_betweenness_centrality()
        location_factor = 1 + edge_betweenness.get(location, 0.1) * 2
        
        # Calculate final probability
        p_cascade = severity_prob[severity] * time_factor * location_factor
        
        # Ensure probability is in [0, 1]
        return min(max(p_cascade, 0), 1)
    
    def simulate_incident_impact(self, location, severity='major', duration=None):
        """
        Simulate the impact of an incident.
        
        Args:
            location: Edge tuple (u, v)
            severity: Incident severity
            duration: Incident duration in minutes (if None, use average)
        
        Returns:
            Dict with impact metrics
        """
        if duration is None:
            duration = self.AVG_CLEARANCE_TIME * (1 + np.random.normal(0, 0.2))
        
        # Get edge attributes
        edge_data = self.G[location[0]][location[1]]
        capacity = edge_data['capacity']
        distance = edge_data['distance']
        
        # Calculate capacity reduction
        capacity_reduction = {
            'minor': 0.5,
            'major': 0.75,
            'fatal': 1.0
        }
        
        remaining_capacity = capacity * (1 - capacity_reduction[severity])
        
        # Calculate queue length using deterministic queuing theory
        arrival_rate = capacity * 0.87  # Current utilization
        if arrival_rate > remaining_capacity:
            queue_growth_rate = arrival_rate - remaining_capacity  # veh/hr
            max_queue = queue_growth_rate * duration / 60  # vehicles
            
            # Spatial extent (assuming 7m per vehicle)
            queue_length_km = max_queue * 0.007
        else:
            queue_length_km = 0
        
        # Calculate delay
        if queue_length_km > 0:
            # Average delay per vehicle (using triangular distribution)
            avg_delay = duration / 2  # minutes
            total_delay_hours = (max_queue * avg_delay) / 60
        else:
            total_delay_hours = 0
        
        # Bidirectional impact (rubbernecking)
        bidirectional_impact = np.random.random() < self.BIDIRECTIONAL_IMPACT_RATE
        if bidirectional_impact:
            # Opposite direction sees capacity reduction
            opposite_location = (location[1], location[0])
            if self.G.has_edge(*opposite_location):
                opposite_capacity = self.G[opposite_location[0]][opposite_location[1]]['capacity']
                opposite_reduction = opposite_capacity * self.RUBBERNECKING_FACTOR
                opposite_delay = opposite_reduction * duration / 60 * 0.5  # vehicle-hours
                total_delay_hours += opposite_delay
        
        # Check for cascade
        time_of_day = np.random.randint(0, 24)
        p_cascade = self.calculate_cascade_probability(location, time_of_day, severity)
        cascade_occurred = np.random.random() < p_cascade
        
        if cascade_occurred:
            # Secondary incident adds more delay
            cascade_delay = total_delay_hours * 0.5  # 50% additional delay
            total_delay_hours += cascade_delay
        
        return {
            'location': location,
            'severity': severity,
            'duration_min': duration,
            'queue_length_km': queue_length_km,
            'total_delay_hours': total_delay_hours,
            'bidirectional_impact': bidirectional_impact,
            'cascade_occurred': cascade_occurred,
            'affected_vehicles': int(max_queue) if queue_length_km > 0 else 0
        }
    
    def run_monte_carlo_simulation(self, n_simulations=1000):
        """Run Monte Carlo simulation of incidents."""
        results = []
        
        # Get all edges
        edges = list(self.G.edges())
        
        # Severity distribution (from historical data)
        severity_weights = [0.7, 0.25, 0.05]  # minor, major, fatal
        severities = ['minor', 'major', 'fatal']
        
        for _ in range(n_simulations):
            # Random incident location
            location = edges[np.random.randint(len(edges))]
            
            # Random severity
            severity = np.random.choice(severities, p=severity_weights)
            
            # Simulate impact
            impact = self.simulate_incident_impact(location, severity)
            results.append(impact)
        
        return pd.DataFrame(results)

# ============================================================================
# RESILIENCE METRICS
# ============================================================================

class ResilienceMetrics:
    """Calculate network resilience metrics."""
    
    def __init__(self, graph, simulation_results):
        self.G = graph
        self.results = simulation_results
        
    def calculate_recovery_time(self, percentile=90):
        """Calculate recovery time distribution."""
        recovery_times = self.results['duration_min'].values
        return {
            'mean': np.mean(recovery_times),
            'median': np.median(recovery_times),
            'p90': np.percentile(recovery_times, percentile),
            'std': np.std(recovery_times)
        }
    
    def calculate_network_efficiency(self):
        """Calculate network efficiency loss due to incidents."""
        # Baseline efficiency (convert to undirected for calculation)
        G_undirected = self.G.to_undirected()
        baseline_efficiency = nx.global_efficiency(G_undirected)
        
        # Calculate efficiency loss for each incident type
        efficiency_loss = {}
        
        for severity in ['minor', 'major', 'fatal']:
            severity_incidents = self.results[self.results['severity'] == severity]
            if len(severity_incidents) > 0:
                avg_delay = severity_incidents['total_delay_hours'].mean()
                # Normalize by network size
                normalized_loss = avg_delay / (len(self.G.nodes()) * 100)
                efficiency_loss[severity] = normalized_loss
        
        return {
            'baseline': baseline_efficiency,
            'loss_by_severity': efficiency_loss
        }
    
    def validate_against_observed(self):
        """Validate simulation against observed statistics."""
        # Cascade rate
        simulated_cascade_rate = self.results['cascade_occurred'].mean()
        observed_cascade_rate = 0.111
        cascade_error = abs(simulated_cascade_rate - observed_cascade_rate) / observed_cascade_rate
        
        # Bidirectional impact rate
        simulated_bidirectional = self.results['bidirectional_impact'].mean()
        observed_bidirectional = 0.33
        bidirectional_error = abs(simulated_bidirectional - observed_bidirectional) / observed_bidirectional
        
        return {
            'cascade_rate': {
                'simulated': simulated_cascade_rate,
                'observed': observed_cascade_rate,
                'error': cascade_error
            },
            'bidirectional_rate': {
                'simulated': simulated_bidirectional,
                'observed': observed_bidirectional,
                'error': bidirectional_error
            },
            'validation_passed': cascade_error < 0.1 and bidirectional_error < 0.1
        }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_network_centrality_plot(network, analyzer):
    """Create network graph with centrality visualization."""
    print("Generating network centrality plot...")
    
    G = network.get_graph()
    node_betweenness, edge_betweenness = analyzer.calculate_betweenness_centrality()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Subplot 1: Network topology with node centrality
    pos = nx.get_node_attributes(G, 'pos')
    
    # Node colors based on betweenness centrality
    node_colors = [node_betweenness[node] for node in G.nodes()]
    
    # Node sizes based on type
    node_types = nx.get_node_attributes(G, 'type')
    node_sizes = []
    for node in G.nodes():
        if node_types[node] == 'capital':
            node_sizes.append(1000)
        elif node_types[node] == 'border':
            node_sizes.append(700)
        elif node_types[node] == 'city':
            node_sizes.append(500)
        else:
            node_sizes.append(300)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          cmap='YlOrRd', vmin=0, vmax=max(node_colors),
                          ax=ax1, alpha=0.9)
    
    # Draw edges with width based on capacity
    edges = G.edges()
    capacities = [G[u][v]['capacity'] for u, v in edges]
    edge_widths = [c/2000 for c in capacities]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', ax=ax1)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax1)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', 
                               norm=plt.Normalize(vmin=0, vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, label='Betweenness Centrality')
    
    ax1.set_title('Highway Network Topology and Node Centrality', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Add legend for node types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, label='Capital'),
        Patch(facecolor='orange', alpha=0.5, label='Border Crossing'),
        Patch(facecolor='yellow', alpha=0.5, label='City'),
        Patch(facecolor='lightgray', alpha=0.5, label='Junction')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    # Subplot 2: Critical segments bar chart
    vulnerability = analyzer.calculate_vulnerability_index()
    critical_segments = analyzer.identify_critical_segments(top_n=10)
    
    segment_names = []
    vulnerability_scores = []
    
    for (u, v), metrics in critical_segments:
        highway = G[u][v].get('highway', 'Unknown')
        segment_names.append(f"{u[:8]}-{v[:8]}\n({highway})")
        vulnerability_scores.append(metrics['total_score'])
    
    y_pos = np.arange(len(segment_names))
    colors = ['red' if score > 0.1 else 'orange' if score > 0.05 else 'yellow' 
             for score in vulnerability_scores]
    
    bars = ax2.barh(y_pos, vulnerability_scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(segment_names, fontsize=8)
    ax2.set_xlabel('Vulnerability Score', fontsize=10)
    ax2.set_title('Top 10 Critical Network Segments', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, score in zip(bars, vulnerability_scores):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=8)
    
    plt.suptitle('Slovenia Highway Network Vulnerability Analysis', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'fig_17_network_graph_centrality.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def create_cascade_probability_heatmap(model, network):
    """Create heatmap of cascade probability by location and time."""
    print("Generating cascade probability heatmap...")
    
    G = network.get_graph()
    edges = list(G.edges())[:15]  # Top 15 edges for visualization
    hours = range(24)
    severities = ['minor', 'major', 'fatal']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, severity in enumerate(severities):
        # Calculate cascade probabilities
        prob_matrix = np.zeros((len(edges), len(hours)))
        
        for i, edge in enumerate(edges):
            for j, hour in enumerate(hours):
                prob = model.calculate_cascade_probability(edge, hour, severity)
                prob_matrix[i, j] = prob
        
        # Create heatmap
        ax = axes[idx]
        im = ax.imshow(prob_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.5)
        
        # Set labels
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], rotation=45)
        ax.set_yticks(range(len(edges)))
        
        edge_labels = [f"{e[0][:5]}-{e[1][:5]}" for e in edges]
        ax.set_yticklabels(edge_labels, fontsize=8)
        
        ax.set_xlabel('Hour of Day', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Highway Segment', fontsize=10)
        ax.set_title(f'{severity.capitalize()} Incidents', fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='P(Cascade)')
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(edges), 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.3)
    
    plt.suptitle('Cascade Probability by Location, Time, and Severity', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'fig_18_cascade_probability_heatmap.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def create_resilience_metrics_plot(metrics, simulation_results):
    """Create resilience metrics visualization."""
    print("Generating resilience metrics plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Subplot 1: Recovery time distribution
    ax1 = axes[0, 0]
    recovery_times = simulation_results['duration_min'].values
    
    counts, bins, patches = ax1.hist(recovery_times, bins=30, density=True,
                                     alpha=0.7, color='steelblue', edgecolor='black')
    
    # Fit and plot gamma distribution
    alpha, loc, beta = stats.gamma.fit(recovery_times)
    x = np.linspace(recovery_times.min(), recovery_times.max(), 100)
    ax1.plot(x, stats.gamma.pdf(x, alpha, loc, beta), 'r-', linewidth=2,
            label=f'Gamma fit: α={alpha:.2f}, β={beta:.2f}')
    
    # Add vertical lines for key percentiles
    recovery_stats = metrics.calculate_recovery_time()
    ax1.axvline(recovery_stats['mean'], color='green', linestyle='--', 
               label=f"Mean: {recovery_stats['mean']:.1f} min")
    ax1.axvline(recovery_stats['p90'], color='orange', linestyle='--',
               label=f"90th %ile: {recovery_stats['p90']:.1f} min")
    
    ax1.set_xlabel('Recovery Time (minutes)', fontsize=10)
    ax1.set_ylabel('Probability Density', fontsize=10)
    ax1.set_title('Incident Recovery Time Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Delay by severity
    ax2 = axes[0, 1]
    severity_delays = simulation_results.groupby('severity')['total_delay_hours'].agg(['mean', 'std'])
    
    severities = severity_delays.index
    x_pos = np.arange(len(severities))
    means = severity_delays['mean'].values
    stds = severity_delays['std'].values
    
    colors = ['yellow', 'orange', 'red']
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s.capitalize() for s in severities])
    ax2.set_ylabel('Average Total Delay (vehicle-hours)', fontsize=10)
    ax2.set_title('Impact by Incident Severity', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 3: Cascade and bidirectional rates
    ax3 = axes[1, 0]
    
    validation = metrics.validate_against_observed()
    
    categories = ['Cascade Rate', 'Bidirectional Impact']
    simulated = [validation['cascade_rate']['simulated'], 
                validation['bidirectional_rate']['simulated']]
    observed = [validation['cascade_rate']['observed'],
               validation['bidirectional_rate']['observed']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, simulated, width, label='Simulated',
                   color='lightblue', edgecolor='black')
    bars2 = ax3.bar(x + width/2, observed, width, label='Observed',
                   color='lightcoral', edgecolor='black')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel('Rate', fontsize=10)
    ax3.set_title('Model Validation: Simulated vs Observed', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 4: Queue length distribution
    ax4 = axes[1, 1]
    
    queue_lengths = simulation_results[simulation_results['queue_length_km'] > 0]['queue_length_km']
    
    if len(queue_lengths) > 0:
        ax4.hist(queue_lengths, bins=30, color='darkred', alpha=0.7, edgecolor='black')
        
        # Add statistics
        mean_queue = queue_lengths.mean()
        max_queue = queue_lengths.max()
        ax4.axvline(mean_queue, color='blue', linestyle='--',
                   label=f'Mean: {mean_queue:.2f} km')
        ax4.axvline(max_queue, color='red', linestyle='--',
                   label=f'Max: {max_queue:.2f} km')
        
        ax4.set_xlabel('Queue Length (km)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('Queue Length Distribution (when formed)', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Incidents with queues: {len(queue_lengths)}/{len(simulation_results)}\n'
        stats_text += f'Avg affected vehicles: {simulation_results["affected_vehicles"].mean():.0f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.suptitle('Network Resilience Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'fig_19_resilience_metrics.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("NETWORK RESILIENCE MODEL - CASCADE EFFECTS ANALYSIS")
    print("Task 4.3: Quantify Network Vulnerability")
    print("="*60 + "\n")
    
    # Build network
    print("Building highway network graph...")
    network = SloveniaHighwayNetwork()
    G = network.get_graph()
    print(f"  Nodes: {len(G.nodes())}")
    print(f"  Edges: {len(G.edges())}")
    
    # Analyze vulnerability
    print("\nAnalyzing network vulnerability...")
    analyzer = NetworkVulnerabilityAnalyzer(G)
    node_betweenness, edge_betweenness = analyzer.calculate_betweenness_centrality()
    critical_segments = analyzer.identify_critical_segments(top_n=5)
    
    print("\nTop 5 Critical Segments:")
    for (u, v), metrics in critical_segments:
        highway = G[u][v].get('highway', 'Unknown')
        print(f"  {u} → {v} ({highway})")
        print(f"    Vulnerability Score: {metrics['total_score']:.3f}")
        print(f"    Betweenness: {metrics['betweenness']:.3f}")
    
    # Run cascade simulation
    print("\nRunning cascade propagation simulation...")
    model = CascadePropagationModel(G)
    simulation_results = model.run_monte_carlo_simulation(n_simulations=5000)
    
    print(f"  Simulations complete: {len(simulation_results)} incidents")
    
    # Calculate metrics
    print("\nCalculating resilience metrics...")
    metrics = ResilienceMetrics(G, simulation_results)
    
    recovery_stats = metrics.calculate_recovery_time()
    print(f"\nRecovery Time Statistics:")
    print(f"  Mean: {recovery_stats['mean']:.1f} minutes")
    print(f"  Median: {recovery_stats['median']:.1f} minutes")
    print(f"  90th percentile: {recovery_stats['p90']:.1f} minutes")
    
    efficiency = metrics.calculate_network_efficiency()
    print(f"\nNetwork Efficiency:")
    print(f"  Baseline efficiency: {efficiency['baseline']:.3f}")
    for severity, loss in efficiency['loss_by_severity'].items():
        print(f"  {severity.capitalize()} incident loss: {loss:.4f}")
    
    # Validate against observed data
    print("\nValidating against observed statistics...")
    validation = metrics.validate_against_observed()
    
    print(f"\nCascade Rate:")
    print(f"  Simulated: {validation['cascade_rate']['simulated']:.1%}")
    print(f"  Observed: {validation['cascade_rate']['observed']:.1%}")
    print(f"  Error: {validation['cascade_rate']['error']:.1%}")
    
    print(f"\nBidirectional Impact Rate:")
    print(f"  Simulated: {validation['bidirectional_rate']['simulated']:.1%}")
    print(f"  Observed: {validation['bidirectional_rate']['observed']:.1%}")
    print(f"  Error: {validation['bidirectional_rate']['error']:.1%}")
    
    if validation['validation_passed']:
        print("\n✓ Model validation PASSED (errors < 10%)")
    else:
        print("\n✗ Model validation FAILED (errors > 10%)")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_network_centrality_plot(network, analyzer)
    create_cascade_probability_heatmap(model, network)
    create_resilience_metrics_plot(metrics, simulation_results)
    
    # Save results to JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'network': {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'avg_degree': sum(dict(G.degree()).values()) / len(G.nodes())
        },
        'critical_segments': [
            {
                'segment': f"{u}-{v}",
                'highway': G[u][v].get('highway', 'Unknown'),
                'vulnerability_score': metrics['total_score'],
                'betweenness': metrics['betweenness']
            }
            for (u, v), metrics in critical_segments
        ],
        'simulation': {
            'n_incidents': len(simulation_results),
            'cascade_rate': float(simulation_results['cascade_occurred'].mean()),
            'bidirectional_rate': float(simulation_results['bidirectional_impact'].mean()),
            'avg_delay_hours': float(simulation_results['total_delay_hours'].mean()),
            'avg_queue_km': float(simulation_results['queue_length_km'].mean())
        },
        'recovery': {
            'mean_minutes': recovery_stats['mean'],
            'median_minutes': recovery_stats['median'],
            'p90_minutes': recovery_stats['p90']
        },
        'validation': {
            'cascade_error': float(validation['cascade_rate']['error']),
            'bidirectional_error': float(validation['bidirectional_rate']['error']),
            'passed': bool(validation['validation_passed'])
        }
    }
    
    output_file = REPORTS_DIR / 'network_resilience_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*60)
    print("NETWORK RESILIENCE ANALYSIS COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()