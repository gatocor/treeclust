"""
Example usage of the new consensus matrix-based clustering functionality.

This example demonstrates how to use the ConsensusClusteringLeiden class
to perform robust clustering with consensus matrix generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from treeclust import ConsensusClusteringLeiden
from scipy.sparse import csr_matrix

def create_example_data():
    """Create a simple example connectivity matrix for testing."""
    np.random.seed(42)
    
    # Create a simple block connectivity matrix
    # 3 blocks of sizes 10, 15, 8
    block_sizes = [10, 15, 8]
    n_total = sum(block_sizes)
    
    # Create adjacency matrix
    adj_matrix = np.zeros((n_total, n_total))
    
    start_idx = 0
    for block_size in block_sizes:
        end_idx = start_idx + block_size
        
        # High connectivity within block
        for i in range(start_idx, end_idx):
            for j in range(start_idx, end_idx):
                if i != j:
                    adj_matrix[i, j] = np.random.uniform(0.7, 1.0)
        
        # Lower connectivity between blocks
        for i in range(start_idx, end_idx):
            for j in range(n_total):
                if j < start_idx or j >= end_idx:
                    adj_matrix[i, j] = np.random.uniform(0.0, 0.2)
        
        start_idx = end_idx
    
    # Make symmetric
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    
    return csr_matrix(adj_matrix)

def main():
    """Main example function."""
    print("Creating example connectivity matrix...")
    connectivity_matrix = create_example_data()
    
    print(f"Connectivity matrix shape: {connectivity_matrix.shape}")
    
    # Initialize consensus clustering
    print("\\nInitializing consensus clustering with hierarchical clustering...")
    consensus_leiden = ConsensusClusteringLeiden(
        connectivity_matrix=connectivity_matrix.toarray(),
        parameter_range=(0.1, 2.0),  # Will be automatically expanded
        n_clusters_max=10,
        random_state=42,
        n_iter=20,  # Number of runs per parameter
        consensus_threshold=0.6,
        hierarchical_method='ward',  # Use Ward linkage for hierarchical clustering
        hierarchical_criterion='distance',  # Use distance criterion
        n_clusters=None,  # Let it determine automatically
        verbose=True
    )
    
    # Fit the model
    print("\\nFitting consensus clustering...")
    consensus_leiden.fit(verbose=True)
    
    # Get results
    final_labels = consensus_leiden.predict()
    consensus_matrix = consensus_leiden.get_consensus_matrix()
    
    print(f"\\nResults:")
    print(f"Number of final clusters: {len(np.unique(final_labels))}")
    print(f"Cluster sizes: {np.bincount(final_labels)}")
    
    # Compute stability metrics
    stability_metrics = consensus_leiden.compute_stability_metrics()
    print(f"\\nStability metrics:")
    for metric, value in stability_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot consensus matrix
    print("\\nPlotting consensus matrix...")
    fig, ax = consensus_leiden.plot_consensus_matrix(figsize=(8, 6))
    plt.savefig('consensus_matrix_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot dendrogram from hierarchical clustering
    print("Plotting hierarchical clustering dendrogram...")
    try:
        fig, ax = consensus_leiden.plot_dendrogram(figsize=(12, 8))
        plt.savefig('dendrogram_example.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Show different cluster levels
        print("\\nExploring different cluster levels:")
        for n_clust in [2, 3, 4, 5]:
            labels_at_level = consensus_leiden.get_clusters_at_level(n_clust)
            print(f"  {n_clust} clusters: sizes = {np.bincount(labels_at_level)}")
            
    except Exception as e:
        print(f"Could not plot dendrogram or explore cluster levels: {e}")
    
    # Plot stability by parameter
    print("\\nPlotting stability by parameter...")
    fig, axes = consensus_leiden.plot_stability_by_parameter(figsize=(12, 5))
    plt.savefig('stability_by_parameter_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Try stability selection
    print("\\nTrying stability-based parameter selection with hierarchical clustering...")
    consensus_leiden_stable = ConsensusClusteringLeiden(
        connectivity_matrix=connectivity_matrix.toarray(),
        parameter_range=(0.1, 2.0),
        n_clusters_max=10,
        random_state=42,
        n_iter=15,
        consensus_threshold=0.6,
        hierarchical_method='complete',  # Try different method
        n_clusters=3,  # Force 3 clusters this time
        verbose=False
    )
    
    consensus_leiden_stable.fit_with_stability_selection(
        stability_threshold=0.7,
        verbose=True
    )
    
    stable_labels = consensus_leiden_stable.predict()
    print(f"\\nStability-selected results:")
    print(f"Number of clusters: {len(np.unique(stable_labels))}")
    print(f"Cluster sizes: {np.bincount(stable_labels)}")
    
    # Get parameter stability scores
    stability_scores = consensus_leiden.get_parameter_stability_scores()
    print(f"\\nParameter stability scores:")
    for param, score in sorted(stability_scores.items()):
        print(f"  Parameter {param:.4f}: {score:.4f}")
    
    print("\\nExample completed successfully!")

if __name__ == "__main__":
    main()