"""
Simple step-by-step demonstration of consensus matrix clustering.

This script shows the basic workflow for using consensus matrix clustering
with hierarchical clustering on synthetic 2D data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from treeclust import ConsensusClusteringLeiden

def main():
    print("=== Consensus Matrix Clustering Demo ===\\n")
    
    # Step 1: Create synthetic data
    print("Step 1: Creating synthetic 2D data with 3 clusters...")
    X, y_true = make_blobs(n_samples=90, centers=3, cluster_std=1.2, random_state=42)
    print(f"  Created {X.shape[0]} points in 2D")
    print(f"  True cluster sizes: {np.bincount(y_true)}")
    
    # Step 2: Build connectivity matrix
    print("\\nStep 2: Building connectivity matrix...")
    connectivity = kneighbors_graph(X, n_neighbors=6, mode='connectivity', include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)  # Make symmetric
    print(f"  Connectivity matrix shape: {connectivity.shape}")
    print(f"  Number of edges: {connectivity.nnz // 2}")  # Divide by 2 since symmetric
    
    # Step 3: Initialize consensus clustering
    print("\\nStep 3: Initializing consensus clustering...")
    consensus_model = ConsensusClusteringLeiden(
        connectivity_matrix=connectivity.toarray(),
        parameter_range=(0.1, 0.8),  # Parameter range for Leiden
        n_iter=20,                   # Number of runs per parameter
        hierarchical_method='ward',   # Hierarchical clustering method
        n_clusters=3,                # Target number of clusters
        random_state=42
    )
    print("  ✓ Model initialized")
    
    # Step 4: Fit the model
    print("\\nStep 4: Fitting consensus clustering...")
    consensus_model.fit(verbose=False)
    print("  ✓ Model fitted")
    
    # Step 5: Get results
    print("\\nStep 5: Getting results...")
    y_pred = consensus_model.predict()
    consensus_matrix = consensus_model.get_consensus_matrix()
    print(f"  Predicted cluster sizes: {np.bincount(y_pred)}")
    print(f"  Consensus matrix shape: {consensus_matrix.shape}")
    
    # Step 6: Evaluate results
    print("\\nStep 6: Evaluating clustering quality...")
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(y_true, y_pred)
    print(f"  Adjusted Rand Index: {ari:.3f}")
    
    # Calculate consensus matrix statistics
    consensus_mean = np.mean(consensus_matrix[np.triu_indices_from(consensus_matrix, k=1)])
    consensus_std = np.std(consensus_matrix[np.triu_indices_from(consensus_matrix, k=1)])
    print(f"  Mean consensus value: {consensus_mean:.3f}")
    print(f"  Consensus std: {consensus_std:.3f}")
    
    # Step 7: Visualize results
    print("\\nStep 7: Creating visualizations...")
    
    # Plot the clustering results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original data
    axes[0].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.7)
    axes[0].set_title('Original Data')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    
    # True clusters
    scatter1 = axes[1].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10')
    axes[1].set_title(f'True Clusters ({len(np.unique(y_true))} clusters)')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    
    # Predicted clusters
    scatter2 = axes[2].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10')
    axes[2].set_title(f'Predicted Clusters ({len(np.unique(y_pred))} clusters)')
    axes[2].set_xlabel('X1')
    axes[2].set_ylabel('X2')
    
    plt.tight_layout()
    plt.savefig('demo_clustering_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot consensus matrix
    fig, ax = consensus_model.plot_consensus_matrix(figsize=(6, 5))
    plt.savefig('demo_consensus_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Saved visualizations:")
    print("    - demo_clustering_results.png")
    print("    - demo_consensus_matrix.png")
    
    # Step 8: Explore hierarchical structure
    print("\\nStep 8: Exploring hierarchical clustering...")
    print("  Different numbers of clusters:")
    for n in [2, 3, 4, 5]:
        labels_n = consensus_model.get_clusters_at_level(n)
        ari_n = adjusted_rand_score(y_true, labels_n)
        print(f"    {n} clusters: sizes = {list(np.bincount(labels_n))}, ARI = {ari_n:.3f}")
    
    # Step 9: Summary
    print("\\n=== Summary ===")
    print(f"✓ Successfully clustered {X.shape[0]} points")
    print(f"✓ Found {len(np.unique(y_pred))} clusters (expected {len(np.unique(y_true))})")
    print(f"✓ Achieved ARI of {ari:.3f}")
    print(f"✓ Consensus matrix shows clustering stability")
    print("\\nThe consensus matrix approach with hierarchical clustering provides:")
    print("  • Robust clustering through multiple runs")
    print("  • Hierarchical structure for exploring different cluster numbers")
    print("  • Stability assessment through consensus values")
    print("  • Visual interpretation through dendrograms and consensus matrices")

if __name__ == "__main__":
    main()