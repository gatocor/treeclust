"""
Example demonstrating hierarchy graph visualization with igraph.

This example shows how to:
1. Create a coassociation matrix from ensemble clustering
2. Perform hierarchical clustering with stability analysis  
3. Create and visualize the hierarchy graph with igraph
4. Plot nodes colored by stability scores
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from treeclust.neighbors.coassotiation import CoassociationDistanceMatrix
from treeclust.bootstrapping.parameter_bootstrapper import ParameterBootstrapper

# Check if igraph is available
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
    print("igraph is available for graph visualization")
except ImportError:
    IGRAPH_AVAILABLE = False
    print("igraph not available - will use dictionary representation")


def create_sample_data():
    """Create sample data for clustering."""
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=150, centers=4, n_features=2, 
                          cluster_std=1.5, random_state=42)
    return X, y_true


def plot_hierarchy_graph_igraph(graph, title="Hierarchy Graph", figsize=(12, 8)):
    """
    Plot hierarchy graph using igraph's built-in plotting.
    
    Parameters
    ----------
    graph : igraph.Graph
        The hierarchy graph to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    if not IGRAPH_AVAILABLE:
        print("igraph not available for plotting")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Set up layout - use sugiyama for hierarchical directed graphs
    # This works better for DAGs that might not be strict trees
    try:
        layout = graph.layout("sugiyama")  # Hierarchical layout for directed graphs
    except:
        # Fallback to fruchterman_reingold if sugiyama fails
        try:
            layout = graph.layout("fruchterman_reingold")
        except:
            # Last resort: use kamada_kawai
            layout = graph.layout("kamada_kawai")
    
    # Color nodes by stability score
    stabilities = graph.vs['stability']
    max_stability = max(stabilities) if stabilities else 1.0
    min_stability = min(stabilities) if stabilities else 0.0
    
    # Normalize stability scores to [0, 1] for coloring
    if max_stability > min_stability:
        normalized_stabilities = [(s - min_stability) / (max_stability - min_stability) 
                                 for s in stabilities]
    else:
        normalized_stabilities = [0.5] * len(stabilities)
    
    # Create colors - convert numpy arrays to tuples for igraph
    color_array = plt.cm.viridis(normalized_stabilities)
    colors = [tuple(color) for color in color_array]
    
    # Size nodes by cluster size
    sizes = graph.vs['size']
    max_size = max(sizes) if sizes else 1
    node_sizes = [max(20, min(100, 30 * s / max_size)) for s in sizes]
    
    # Plot using igraph
    visual_style = {
        "vertex_color": colors,
        "vertex_size": node_sizes,
        "vertex_label": [f"L{v['level']}_C{v['cluster_id']}" for v in graph.vs],
        "vertex_label_size": 8,
        "edge_arrow_size": 0.5,
        "edge_width": [2 * w for w in graph.es['weight']],
        "layout": layout,
        "bbox": (800, 600),
        "margin": 50
    }
    
    # Save plot
    ig.plot(graph, f"hierarchy_graph_{title.lower().replace(' ', '_')}.png", **visual_style)
    print(f"Saved igraph plot as hierarchy_graph_{title.lower().replace(' ', '_')}.png")


def plot_hierarchy_graph_matplotlib(graph, title="Hierarchy Graph", figsize=(12, 8)):
    """
    Plot hierarchy graph using matplotlib with manual positioning.
    
    Parameters
    ----------
    graph : igraph.Graph or dict
        The hierarchy graph to plot
    title : str
        Plot title  
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if isinstance(graph, dict):
        # Handle dictionary representation
        nodes = graph['nodes']
        edges = graph['edges']
        
        # Group nodes by level
        levels = {}
        for node in nodes:
            level = node['level']
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        # Position nodes
        node_positions = {}
        level_height = 1.0 / (len(levels) + 1)
        
        for level_idx, (level, level_nodes) in enumerate(sorted(levels.items())):
            y = 1.0 - (level_idx + 1) * level_height
            node_width = 1.0 / (len(level_nodes) + 1)
            
            for node_idx, node in enumerate(level_nodes):
                x = (node_idx + 1) * node_width
                node_positions[node['id']] = (x, y)
        
        # Plot edges
        for edge in edges:
            start_pos = node_positions[edge['source']]
            end_pos = node_positions[edge['target']]
            
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', 
                                     alpha=0.6, 
                                     lw=edge['weight'] * 3,
                                     color='gray'))
        
        # Plot nodes
        for node in nodes:
            pos = node_positions[node['id']]
            stability = node['stability']
            size = node['size']
            
            # Color by stability
            color = plt.cm.viridis(stability)
            
            # Size by cluster size
            node_size = max(100, min(1000, 200 * size / max(n['size'] for n in nodes)))
            
            ax.scatter(pos[0], pos[1], s=node_size, c=[color], 
                      alpha=0.8, edgecolors='black', linewidth=1)
            
            # Add label
            ax.text(pos[0], pos[1], f"L{node['level']}_C{node['cluster_id']}", 
                   ha='center', va='center', fontsize=8, weight='bold')
        
    else:
        # Handle igraph.Graph
        if not IGRAPH_AVAILABLE:
            print("Cannot plot igraph.Graph without igraph library")
            return
            
        # Use igraph layout - use sugiyama for hierarchical directed graphs
        try:
            layout = graph.layout("sugiyama")
        except:
            # Fallback to fruchterman_reingold if sugiyama fails
            try:
                layout = graph.layout("fruchterman_reingold")
            except:
                # Last resort: use kamada_kawai
                layout = graph.layout("kamada_kawai")
        
        # Extract positions
        node_positions = {i: (pos[0], pos[1]) for i, pos in enumerate(layout)}
        
        # Plot edges
        for edge in graph.es:
            source_idx = edge.source
            target_idx = edge.target
            start_pos = node_positions[source_idx]
            end_pos = node_positions[target_idx]
            
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', 
                                     alpha=0.6, 
                                     lw=edge['weight'] * 3,
                                     color='gray'))
        
        # Plot nodes
        stabilities = graph.vs['stability']
        sizes = graph.vs['size']
        max_size = max(sizes) if sizes else 1
        
        for i, vertex in enumerate(graph.vs):
            pos = node_positions[i]
            stability = vertex['stability']
            size = vertex['size']
            
            # Color by stability
            color = plt.cm.viridis(stability)
            
            # Size by cluster size
            node_size = max(100, min(1000, 200 * size / max_size))
            
            ax.scatter(pos[0], pos[1], s=node_size, c=[color], 
                      alpha=0.8, edgecolors='black', linewidth=1)
            
            # Add label
            ax.text(pos[0], pos[1], f"L{vertex['level']}_C{vertex['cluster_id']}", 
                   ha='center', va='center', fontsize=8, weight='bold')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel('Cluster Position')
    ax.set_ylabel('Hierarchy Level')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for stability
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Stability Score', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main example function."""
    print("Creating hierarchy graph visualization example...")
    
    # Create sample data
    print("\n1. Creating sample data...")
    X, y_true = create_sample_data()
    print(f"Created {len(X)} samples with {X.shape[1]} features")
    
    # Create parameter bootstrapper for ensemble clustering
    print("\n2. Setting up ensemble clustering...")
    param_ranges = {
        'n_clusters': [3, 4, 5, 6],
        'init': ['k-means++', 'random'],
        'n_init': [10, 20]
    }
    
    param_sampler = ParameterBootstrapper(
        ml_class=KMeans,
        param_ranges=param_ranges,
        n_samples=20,  # Use fewer samples for faster demo
        random_state=42
    )
    
    # Create coassociation matrix
    print("\n3. Creating coassociation matrix...")
    coassoc = CoassociationDistanceMatrix(
        ml_class_list=[param_sampler],
        n_splits=15,  # Use fewer runs for demo
        convergence_threshold=0.001,
        max_runs=50,
        random_state=42
    )
    
    # Fit the model
    print("\n4. Fitting ensemble clustering...")
    coassoc.fit(X)
    print(f"Converged after {coassoc.n_runs_} iterations")
    
    # Perform hierarchical clustering
    print("\n5. Performing hierarchical clustering...")
    hierarchical_result = coassoc.hierarchical_clustering(
        method='ward',
        max_levels=4,
        min_cluster_size=5,
        stability_threshold=0.1
    )
    
    print("Hierarchical clustering results:")
    print(f"- Number of hierarchy levels: {len(hierarchical_result['cluster_assignments'])}")
    print(f"- Linkage matrix shape: {hierarchical_result['linkage_matrix'].shape}")
    print(f"- Hierarchy graph type: {type(hierarchical_result['hierarchy_graph'])}")
    
    # Plot the hierarchy graph
    print("\n6. Plotting hierarchy graph...")
    
    if IGRAPH_AVAILABLE and hasattr(hierarchical_result['hierarchy_graph'], 'vs'):
        print("Using igraph for visualization...")
        
        # Plot with igraph's built-in plotting
        try:
            plot_hierarchy_graph_igraph(
                hierarchical_result['hierarchy_graph'], 
                title="Hierarchy Graph (igraph)"
            )
        except Exception as e:
            print(f"igraph plotting failed: {e}")
            print("Falling back to matplotlib...")
    
    # Always try matplotlib plotting as well
    print("Creating matplotlib visualization...")
    plot_hierarchy_graph_matplotlib(
        hierarchical_result['hierarchy_graph'], 
        title="Clustering Hierarchy Graph"
    )
    
    # Print graph statistics
    graph = hierarchical_result['hierarchy_graph']
    if isinstance(graph, dict):
        print(f"\nGraph statistics (dict format):")
        print(f"- Number of nodes: {len(graph['nodes'])}")
        print(f"- Number of edges: {len(graph['edges'])}")
        print(f"- Number of levels: {len(set(n['level'] for n in graph['nodes']))}")
    elif IGRAPH_AVAILABLE and hasattr(graph, 'vs'):
        print(f"\nGraph statistics (igraph format):")
        print(f"- Number of vertices: {graph.vcount()}")
        print(f"- Number of edges: {graph.ecount()}")
        print(f"- Number of levels: {len(set(graph.vs['level']))}")
        print(f"- Average stability: {np.mean(graph.vs['stability']):.3f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()