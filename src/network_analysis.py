"""
Network Graph Analysis for Slovenian Highway System
Analyzes network topology, centrality, and vulnerability to disruptions
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class HighwayNetworkAnalyzer:
    """
    Analyzes the Slovenian highway network as a graph structure
    """
    
    def __init__(self, speed_file, baseline_stats_file):
        """
        Initialize network analyzer
        
        Args:
            speed_file: Path to vehicle speed data
            baseline_stats_file: Path to baseline statistics
        """
        self.speed_file = speed_file
        self.baseline_stats_file = baseline_stats_file
        self.G = None
        self.segment_map = {}
        self.centrality_metrics = {}
        
    def build_network_graph(self):
        """Build the highway network graph from segment data"""
        
        print("Building highway network graph...")
        
        # Create directed graph
        self.G = nx.DiGraph()
        
        # Define major interchanges (nodes) based on Slovenian Road Segment Analysis
        major_nodes = {
            # Ljubljana Ring - Critical Hub
            'Kozarje': {'type': 'interchange', 'importance': 'critical', 'highways': ['A1', 'A2']},
            'Malence': {'type': 'interchange', 'importance': 'critical', 'highways': ['A1', 'A2']},
            'Zadobrova': {'type': 'interchange', 'importance': 'critical', 'highways': ['A1', 'H3']},
            'Koseze': {'type': 'interchange', 'importance': 'critical', 'highways': ['A2', 'H3']},
            
            # Major Cities
            'Ljubljana': {'type': 'city', 'importance': 'high', 'highways': ['A1', 'A2', 'H3']},
            'Maribor': {'type': 'city', 'importance': 'high', 'highways': ['A1', 'A4', 'A5']},
            'Celje': {'type': 'city', 'importance': 'high', 'highways': ['A1']},
            'Kranj': {'type': 'city', 'importance': 'high', 'highways': ['A2']},
            'Novo_Mesto': {'type': 'city', 'importance': 'high', 'highways': ['A2']},
            'Koper': {'type': 'port', 'importance': 'critical', 'highways': ['A1', 'H5']},
            
            # Other Key Interchanges
            'Slivnica': {'type': 'interchange', 'importance': 'high', 'highways': ['A1', 'A4']},
            'Dragucova': {'type': 'interchange', 'importance': 'high', 'highways': ['A1', 'A5']},
            'Razdrto': {'type': 'interchange', 'importance': 'high', 'highways': ['A1', 'H4']},
            'Srmin': {'type': 'interchange', 'importance': 'high', 'highways': ['A1', 'H5']},
            
            # Border Crossings
            'Sentilj': {'type': 'border', 'importance': 'high', 'highways': ['A1'], 'country': 'AT'},
            'Karawanks': {'type': 'border', 'importance': 'high', 'highways': ['A2'], 'country': 'AT'},
            'Obrezje': {'type': 'border', 'importance': 'high', 'highways': ['A2'], 'country': 'HR'},
            'Gruškovje': {'type': 'border', 'importance': 'high', 'highways': ['A4'], 'country': 'HR'},
            'Pince': {'type': 'border', 'importance': 'medium', 'highways': ['A5'], 'country': 'HU'},
            'Fernetici': {'type': 'border', 'importance': 'medium', 'highways': ['A3'], 'country': 'IT'},
            'Vrtojba': {'type': 'border', 'importance': 'medium', 'highways': ['H4'], 'country': 'IT'},
            'Skofije': {'type': 'border', 'importance': 'medium', 'highways': ['H5'], 'country': 'IT'}
        }
        
        # Add nodes with attributes
        for node, attrs in major_nodes.items():
            self.G.add_node(node, **attrs)
            
        # Define edges (road segments) with realistic distances and capacities
        edges = [
            # A1 Motorway (Sentilj - Koper)
            ('Sentilj', 'Maribor', {'highway': 'A1', 'distance': 17, 'capacity': 4000}),
            ('Maribor', 'Slivnica', {'highway': 'A1', 'distance': 8, 'capacity': 4000}),
            ('Slivnica', 'Celje', {'highway': 'A1', 'distance': 35, 'capacity': 4000}),
            ('Celje', 'Ljubljana', {'highway': 'A1', 'distance': 75, 'capacity': 4000}),
            ('Ljubljana', 'Zadobrova', {'highway': 'A1', 'distance': 5, 'capacity': 5000}),
            ('Zadobrova', 'Malence', {'highway': 'A1', 'distance': 8, 'capacity': 5000}),
            ('Malence', 'Kozarje', {'highway': 'A1', 'distance': 10, 'capacity': 5000}),
            ('Kozarje', 'Razdrto', {'highway': 'A1', 'distance': 45, 'capacity': 3500}),
            ('Razdrto', 'Srmin', {'highway': 'A1', 'distance': 35, 'capacity': 3500}),
            ('Srmin', 'Koper', {'highway': 'A1', 'distance': 5, 'capacity': 3500}),
            
            # A2 Motorway (Karawanks - Obrezje)
            ('Karawanks', 'Kranj', {'highway': 'A2', 'distance': 35, 'capacity': 3500}),
            ('Kranj', 'Ljubljana', {'highway': 'A2', 'distance': 25, 'capacity': 4000}),
            ('Ljubljana', 'Koseze', {'highway': 'A2', 'distance': 5, 'capacity': 5000}),
            ('Koseze', 'Kozarje', {'highway': 'A2', 'distance': 8, 'capacity': 5000}),
            ('Kozarje', 'Malence', {'highway': 'A2', 'distance': 10, 'capacity': 5000}),
            ('Malence', 'Novo_Mesto', {'highway': 'A2', 'distance': 65, 'capacity': 3500}),
            ('Novo_Mesto', 'Obrezje', {'highway': 'A2', 'distance': 40, 'capacity': 3500}),
            
            # A4 Motorway (Slivnica - Gruškovje)
            ('Slivnica', 'Gruškovje', {'highway': 'A4', 'distance': 60, 'capacity': 3000}),
            
            # A5 Motorway (Dragucova - Pince)
            ('Maribor', 'Dragucova', {'highway': 'A1', 'distance': 5, 'capacity': 3500}),
            ('Dragucova', 'Pince', {'highway': 'A5', 'distance': 85, 'capacity': 2500}),
            
            # H3 Northern Ring
            ('Zadobrova', 'Koseze', {'highway': 'H3', 'distance': 12, 'capacity': 4500}),
            
            # H4 Expressway
            ('Razdrto', 'Vrtojba', {'highway': 'H4', 'distance': 40, 'capacity': 2500}),
            
            # H5 Expressway
            ('Srmin', 'Skofije', {'highway': 'H5', 'distance': 10, 'capacity': 3000}),
            
            # A3 to Italy
            ('Kozarje', 'Fernetici', {'highway': 'A3', 'distance': 80, 'capacity': 2500})
        ]
        
        # Add edges to graph
        for u, v, attrs in edges:
            self.G.add_edge(u, v, **attrs)
            # Add reverse direction for most edges (bidirectional highways)
            self.G.add_edge(v, u, **attrs)
            
        print(f"Network graph created:")
        print(f"  Nodes (interchanges/cities): {self.G.number_of_nodes()}")
        print(f"  Edges (road segments): {self.G.number_of_edges()}")
        print(f"  Average degree: {np.mean([d for n, d in self.G.degree()]):.2f}")
        
    def calculate_centrality_metrics(self):
        """Calculate various centrality metrics for network nodes"""
        
        print("\n" + "=" * 60)
        print("CENTRALITY ANALYSIS")
        print("=" * 60)
        
        # Degree centrality
        self.centrality_metrics['degree'] = nx.degree_centrality(self.G)
        
        # Betweenness centrality (nodes that lie on shortest paths)
        self.centrality_metrics['betweenness'] = nx.betweenness_centrality(
            self.G, weight='distance'
        )
        
        # Closeness centrality (average distance to all other nodes)
        self.centrality_metrics['closeness'] = nx.closeness_centrality(
            self.G, distance='distance'
        )
        
        # Eigenvector centrality (importance based on neighbor importance)
        try:
            self.centrality_metrics['eigenvector'] = nx.eigenvector_centrality(
                self.G, max_iter=1000
            )
        except:
            self.centrality_metrics['eigenvector'] = {}
            
        # Load centrality (stress on nodes from shortest paths)
        self.centrality_metrics['load'] = nx.load_centrality(
            self.G, weight='distance'
        )
        
        # Create summary DataFrame
        centrality_df = pd.DataFrame(self.centrality_metrics)
        centrality_df['node_type'] = centrality_df.index.map(
            lambda x: self.G.nodes[x].get('type', 'unknown')
        )
        centrality_df['importance'] = centrality_df.index.map(
            lambda x: self.G.nodes[x].get('importance', 'unknown')
        )
        
        # Calculate composite vulnerability score
        centrality_df['vulnerability_score'] = (
            centrality_df['betweenness'] * 0.4 +
            centrality_df['degree'] * 0.2 +
            centrality_df['load'] * 0.4
        )
        
        # Sort by vulnerability score
        centrality_df = centrality_df.sort_values('vulnerability_score', ascending=False)
        
        print("\nMost Critical Nodes (by Vulnerability Score):")
        print(centrality_df.head(10)[['node_type', 'vulnerability_score', 
                                      'betweenness', 'degree']])
        
        self.centrality_df = centrality_df
        
        return centrality_df
        
    def analyze_network_vulnerability(self):
        """Analyze network vulnerability to node/edge failures"""
        
        print("\n" + "=" * 60)
        print("VULNERABILITY ANALYSIS")
        print("=" * 60)
        
        # Original network metrics
        original_diameter = nx.diameter(self.G.to_undirected())
        original_avg_path = nx.average_shortest_path_length(
            self.G.to_undirected(), weight='distance'
        )
        
        print(f"Original Network Metrics:")
        print(f"  Diameter: {original_diameter}")
        print(f"  Average Path Length: {original_avg_path:.2f} km")
        
        # Test removal of critical nodes
        critical_nodes = ['Kozarje', 'Malence', 'Ljubljana', 'Slivnica']
        
        vulnerability_results = []
        
        for node in critical_nodes:
            # Create copy and remove node
            G_test = self.G.copy()
            G_test.remove_node(node)
            
            # Check if network is still connected
            if nx.is_weakly_connected(G_test):
                try:
                    new_avg_path = nx.average_shortest_path_length(
                        G_test.to_undirected(), weight='distance'
                    )
                    path_increase = (new_avg_path - original_avg_path) / original_avg_path * 100
                except:
                    path_increase = 999  # Network fragmented
            else:
                # Network fragmented
                components = list(nx.weakly_connected_components(G_test))
                path_increase = 999
                
            vulnerability_results.append({
                'node_removed': node,
                'path_length_increase': path_increase,
                'network_fragmented': path_increase == 999
            })
            
        vulnerability_df = pd.DataFrame(vulnerability_results)
        
        print("\nImpact of Node Removal:")
        for _, row in vulnerability_df.iterrows():
            if row['network_fragmented']:
                print(f"  {row['node_removed']:15} → NETWORK FRAGMENTED")
            else:
                print(f"  {row['node_removed']:15} → Path length +{row['path_length_increase']:.1f}%")
                
        # Analyze edge vulnerabilities (road segment failures)
        print("\nCritical Road Segments (by betweenness):")
        edge_betweenness = nx.edge_betweenness_centrality(self.G, weight='distance')
        critical_edges = sorted(edge_betweenness.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        
        for (u, v), score in critical_edges:
            highway = self.G[u][v].get('highway', 'Unknown')
            print(f"  {u:12} → {v:12} ({highway}): {score:.3f}")
            
        self.vulnerability_results = vulnerability_df
        
        return vulnerability_df
        
    def simulate_ripple_effects(self, disrupted_node, disruption_severity=0.5):
        """
        Simulate how a disruption at one node affects the network
        
        Args:
            disrupted_node: Node where disruption occurs
            disruption_severity: Fraction of capacity lost (0-1)
        """
        
        print(f"\n" + "=" * 60)
        print(f"RIPPLE EFFECT SIMULATION: {disrupted_node}")
        print(f"Disruption Severity: {disruption_severity:.0%}")
        print("=" * 60)
        
        if disrupted_node not in self.G.nodes():
            print(f"Node {disrupted_node} not found in network")
            return None
            
        # Calculate shortest paths from disrupted node
        paths = nx.single_source_shortest_path_length(
            self.G, disrupted_node, cutoff=3
        )
        
        # Group nodes by distance (hops)
        distance_groups = defaultdict(list)
        for node, dist in paths.items():
            distance_groups[dist].append(node)
            
        # Simulate impact decay
        impact_by_distance = {}
        base_impact = disruption_severity * 100  # Convert to percentage
        
        for distance, nodes in sorted(distance_groups.items()):
            # Impact decays exponentially with distance
            impact = base_impact * np.exp(-0.5 * distance)
            impact_by_distance[distance] = {
                'nodes': nodes,
                'n_nodes': len(nodes),
                'impact_pct': impact
            }
            
        print("\nRipple Effect Propagation:")
        for dist, data in impact_by_distance.items():
            if dist == 0:
                print(f"  Distance {dist} (disrupted): {data['impact_pct']:.1f}% capacity loss")
            else:
                print(f"  Distance {dist} ({data['n_nodes']} nodes): {data['impact_pct']:.1f}% impact")
                if dist <= 2:
                    print(f"    Affected: {', '.join(data['nodes'][:5])}")
                    
        # Calculate network-wide impact
        total_nodes = self.G.number_of_nodes()
        affected_nodes = sum(data['n_nodes'] for data in impact_by_distance.values())
        network_coverage = affected_nodes / total_nodes * 100
        
        avg_impact = sum(
            data['impact_pct'] * data['n_nodes'] 
            for data in impact_by_distance.values()
        ) / affected_nodes
        
        print(f"\nNetwork-wide Impact:")
        print(f"  Nodes affected: {affected_nodes}/{total_nodes} ({network_coverage:.1f}%)")
        print(f"  Average impact: {avg_impact:.1f}%")
        
        return impact_by_distance
        
    def identify_alternative_routes(self, origin, destination):
        """
        Find alternative routes between two points
        
        Args:
            origin: Start node
            destination: End node
        """
        
        print(f"\n" + "=" * 60)
        print(f"ALTERNATIVE ROUTES: {origin} → {destination}")
        print("=" * 60)
        
        if origin not in self.G or destination not in self.G:
            print("Origin or destination not found in network")
            return None
            
        # Find all simple paths (up to reasonable length)
        try:
            paths = list(nx.all_simple_paths(
                self.G, origin, destination, cutoff=10
            ))
        except nx.NetworkXNoPath:
            print("No path exists between these nodes")
            return None
            
        # Calculate metrics for each path
        route_analysis = []
        
        for i, path in enumerate(paths[:5]):  # Limit to top 5 routes
            total_distance = sum(
                self.G[path[j]][path[j+1]].get('distance', 0)
                for j in range(len(path)-1)
            )
            
            min_capacity = min(
                self.G[path[j]][path[j+1]].get('capacity', 0)
                for j in range(len(path)-1)
            )
            
            highways_used = set(
                self.G[path[j]][path[j+1]].get('highway', 'Unknown')
                for j in range(len(path)-1)
            )
            
            route_analysis.append({
                'route_id': i+1,
                'path': ' → '.join(path),
                'n_segments': len(path)-1,
                'total_distance': total_distance,
                'min_capacity': min_capacity,
                'highways': ', '.join(highways_used)
            })
            
        routes_df = pd.DataFrame(route_analysis)
        
        if len(routes_df) > 0:
            print("\nAvailable Routes:")
            for _, route in routes_df.iterrows():
                print(f"\nRoute {route['route_id']}:")
                print(f"  Path: {route['path']}")
                print(f"  Distance: {route['total_distance']} km")
                print(f"  Min Capacity: {route['min_capacity']} vehicles/hour")
                print(f"  Highways: {route['highways']}")
                
            # Calculate route diversity
            if len(routes_df) > 1:
                avg_distance = routes_df['total_distance'].mean()
                distance_variation = routes_df['total_distance'].std() / avg_distance
                print(f"\nRoute Diversity:")
                print(f"  Number of alternatives: {len(routes_df)}")
                print(f"  Distance variation: {distance_variation:.2%}")
                
                if distance_variation < 0.1:
                    print("  → LOW diversity: Limited alternatives")
                elif distance_variation < 0.3:
                    print("  → MODERATE diversity: Some flexibility")
                else:
                    print("  → HIGH diversity: Good redundancy")
                    
        return routes_df
        
    def visualize_network(self):
        """Create network visualization"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Layout
        pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
        
        # Plot 1: Network structure with node importance
        ax = axes[0]
        
        # Node colors by type
        node_colors = []
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('type', 'unknown')
            if node_type == 'interchange':
                node_colors.append('red')
            elif node_type == 'city':
                node_colors.append('blue')
            elif node_type == 'border':
                node_colors.append('green')
            elif node_type == 'port':
                node_colors.append('orange')
            else:
                node_colors.append('gray')
                
        # Node sizes by betweenness centrality
        if 'betweenness' in self.centrality_metrics:
            node_sizes = [
                1000 * self.centrality_metrics['betweenness'].get(node, 0.1) + 100
                for node in self.G.nodes()
            ]
        else:
            node_sizes = 300
            
        nx.draw_networkx(self.G, pos, ax=ax, 
                        node_color=node_colors,
                        node_size=node_sizes,
                        with_labels=True,
                        font_size=8,
                        edge_color='gray',
                        arrows=False)
        
        ax.set_title('Slovenian Highway Network Structure')
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Interchange'),
            Patch(facecolor='blue', label='City'),
            Patch(facecolor='green', label='Border'),
            Patch(facecolor='orange', label='Port')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Vulnerability heatmap
        ax = axes[1]
        
        if hasattr(self, 'centrality_df'):
            # Create heatmap of centrality metrics
            heatmap_data = self.centrality_df.head(10)[
                ['betweenness', 'degree', 'load', 'vulnerability_score']
            ].T
            
            sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd', 
                       annot=True, fmt='.2f', cbar_kws={'label': 'Score'})
            ax.set_title('Node Vulnerability Metrics (Top 10)')
            ax.set_xlabel('Node')
            ax.set_ylabel('Metric')
            
        plt.tight_layout()
        plt.savefig('./reports/network_analysis.png', dpi=150)
        plt.show()
        
        print("\nNetwork visualization saved to ./reports/network_analysis.png")
        
    def export_network_metrics(self, output_dir='./data'):
        """Export network analysis results"""
        
        # Export centrality metrics
        if hasattr(self, 'centrality_df'):
            centrality_file = f"{output_dir}/network_centrality_metrics.csv"
            self.centrality_df.to_csv(centrality_file)
            print(f"\nCentrality metrics exported to {centrality_file}")
            
        # Export vulnerability results
        if hasattr(self, 'vulnerability_results'):
            vulnerability_file = f"{output_dir}/network_vulnerability.csv"
            self.vulnerability_results.to_csv(vulnerability_file, index=False)
            print(f"Vulnerability analysis exported to {vulnerability_file}")
            
        # Export network summary
        summary = {
            'n_nodes': self.G.number_of_nodes(),
            'n_edges': self.G.number_of_edges(),
            'avg_degree': np.mean([d for n, d in self.G.degree()]),
            'diameter': nx.diameter(self.G.to_undirected()),
            'avg_path_length': nx.average_shortest_path_length(
                self.G.to_undirected(), weight='distance'
            ),
            'density': nx.density(self.G),
            'n_critical_nodes': len(self.centrality_df[
                self.centrality_df['importance'] == 'critical'
            ]) if hasattr(self, 'centrality_df') else 0
        }
        
        summary_df = pd.DataFrame([summary])
        summary_file = f"{output_dir}/network_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Network summary exported to {summary_file}")
        
        return summary


def main():
    """Main execution for network analysis"""
    
    print("\n" + "=" * 60)
    print("NETWORK GRAPH ANALYSIS")
    print("Slovenian Highway System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = HighwayNetworkAnalyzer(
        speed_file='./data/production_merged_vehicle_speed.csv',
        baseline_stats_file='./data/baseline_statistics_2022_2023.csv'
    )
    
    # Build network
    analyzer.build_network_graph()
    
    # Calculate centrality
    centrality_df = analyzer.calculate_centrality_metrics()
    
    # Analyze vulnerability
    vulnerability_df = analyzer.analyze_network_vulnerability()
    
    # Simulate ripple effects from Ljubljana Ring disruption
    print("\n" + "=" * 60)
    print("RIPPLE EFFECT SIMULATIONS")
    print("=" * 60)
    
    # Test disruption at critical interchange
    analyzer.simulate_ripple_effects('Kozarje', disruption_severity=0.7)
    analyzer.simulate_ripple_effects('Malence', disruption_severity=0.5)
    
    # Find alternative routes
    print("\n" + "=" * 60)
    print("ALTERNATIVE ROUTE ANALYSIS")
    print("=" * 60)
    
    # Key routes to analyze
    analyzer.identify_alternative_routes('Koper', 'Sentilj')  # Port to Austria
    analyzer.identify_alternative_routes('Karawanks', 'Obrezje')  # Austria to Croatia
    
    # Visualize network
    analyzer.visualize_network()
    
    # Export results
    summary = analyzer.export_network_metrics()
    
    print("\n" + "=" * 60)
    print("NETWORK ANALYSIS COMPLETE")
    print("=" * 60)
    
    return analyzer, summary


if __name__ == "__main__":
    analyzer, summary = main()