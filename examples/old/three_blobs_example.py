"""
Example usage of consensus matrix clustering with three 2D blobs.

This example creates three clear clusters in 2D space and demonstrates
how the consensus matrix approach with hierarchical clustering works.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.da    for param, score in sorted(stability_scores.items()):
        print(f"  Parameter {param:.4f}: {score:.4f}")
    
    print("\n" + "="*80)
    print("PARTITION RELIABILITY ANALYSIS")
    print("="*80)
    
    # Test partition reliability evaluation
    print("\nEvaluating partition reliability...")
    evaluations = consensus_leiden.evaluate_all_partitions()
    
    # Display evaluation summary
    summary = consensus_leiden.get_partition_reliability_summary()
    print(f"\nReliability Summary:")
    print(f"Total partitions evaluated: {summary['total_partitions']}")
    
    for metric, stats in summary.items():
        if metric not in ['total_partitions', 'reliable_partitions', 'reliability_ratio'] and stats:
            print(f"{metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # Select reliable partitions
    reliable_indices = consensus_leiden.select_reliable_partitions(
        min_silhouette=0.3,
        min_stability=0.4,
        min_size_balance=0.3,
        top_k=10
    )
    
    print(f"\nSelected {len(reliable_indices)} reliable partitions out of {summary['total_partitions']}")
    print(f"Reliability ratio: {summary['reliability_ratio']:.3f}")
    
    # Plot partition reliability
    try:
        consensus_leiden.plot_partition_reliability(figsize=(14, 10), save_path='three_blobs_partition_reliability.png')
        print("Generated three_blobs_partition_reliability.png")
    except Exception as e:
        print(f"Could not plot partition reliability: {e}")
    
    print("\n" + "="*80)
    print("CONSENSUS CLUSTERING WITH PARTITION SELECTION")
    print("="*80)
    
    # Test fit_with_partition_selection
    print("\nFitting with automatic partition selection...")
    consensus_leiden_selective = ConsensusClusteringLeiden(
        connectivity_matrix=connectivity.toarray(),
        parameter_range=np.linspace(0.05, 1.0, 12),
        n_clusters=3,
        n_iter=15,
        random_state=42,
        hierarchical_method='stability_aware'
    )
    
    try:
        consensus_leiden_selective.fit_with_partition_selection(
            verbose=True,
            reliability_kwargs={
                'min_silhouette': 0.4,
                'min_stability': 0.5,
                'min_size_balance': 0.4,
                'top_k': 15
            }
        )
        
        # Compare results
        selective_labels = consensus_leiden_selective.predict()
        selective_ari = adjusted_rand_score(y_true, selective_labels)
        selective_silhouette = silhouette_score(X, selective_labels)
        
        print(f"\nComparison of approaches:")
        print(f"Regular consensus clustering:")
        print(f"  ARI: {ari:.3f}")
        print(f"  Silhouette: {sil:.3f}")
        print(f"  Stability: {stability['consensus_mean']:.3f}")
        
        print(f"\nPartition-selective consensus clustering:")
        print(f"  ARI: {selective_ari:.3f}")
        print(f"  Silhouette: {selective_silhouette:.3f}")
        
        # Get selective reliability summary
        selective_summary = consensus_leiden_selective.get_partition_reliability_summary()
        print(f"  Reliable partitions used: {selective_summary['reliable_partitions']}/{selective_summary['total_partitions']}")
        
    except Exception as e:
        print(f"Could not complete partition selection analysis: {e}")
        
    print("\nThree blobs example completed successfully!")ts import make_blobs
from sklearn.neighbors import kneighbors_graph
from treeclust import ConsensusClusteringLeiden
from scipy.sparse import csr_matrix

def create_three_blobs_data(n_samples=150, centers=3, cluster_std=1.5, random_state=42):
    """Create three blob clusters in 2D space and compute connectivity matrix."""
    
    # Generate 2D data with three blobs
    X, y_true = make_blobs(
        n_samples=n_samples, 
        centers=centers, 
        cluster_std=cluster_std,
        center_box=(-10.0, 10.0),
        random_state=random_state
    )
    
    # Create connectivity matrix using k-nearest neighbors
    # This creates edges between nearby points
    connectivity = kneighbors_graph(
        X, 
        n_neighbors=8,  # Connect each point to 8 nearest neighbors
        mode='connectivity',
        include_self=False
    )
    
    # Make symmetric (undirected graph)
    connectivity = 0.5 * (connectivity + connectivity.T)
    
    return X, y_true, connectivity

def plot_results(X, y_true, y_pred, consensus_matrix, title_prefix=""):
    """Plot the original data, true clusters, predicted clusters, and consensus matrix."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot original data points
    axes[0, 0].scatter(X[:, 0], X[:, 1], c='black', alpha=0.6, s=50)
    axes[0, 0].set_title(f'{title_prefix}Original Data Points')
    axes[0, 0].set_xlabel('X1')
    axes[0, 0].set_ylabel('X2')
    
    # Plot true clusters
    scatter1 = axes[0, 1].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=50)
    axes[0, 1].set_title(f'{title_prefix}True Clusters')
    axes[0, 1].set_xlabel('X1')
    axes[0, 1].set_ylabel('X2')
    plt.colorbar(scatter1, ax=axes[0, 1])
    
    # Plot predicted clusters
    scatter2 = axes[1, 0].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='tab10', s=50)
    axes[1, 0].set_title(f'{title_prefix}Predicted Clusters')
    axes[1, 0].set_xlabel('X1')
    axes[1, 0].set_ylabel('X2')
    plt.colorbar(scatter2, ax=axes[1, 0])
    
    # Plot consensus matrix (ordered by true clusters for comparison)
    order = np.argsort(y_true)
    consensus_ordered = consensus_matrix[np.ix_(order, order)]
    im = axes[1, 1].imshow(consensus_ordered, cmap='viridis', aspect='auto')
    axes[1, 1].set_title(f'{title_prefix}Consensus Matrix (ordered by true clusters)')
    axes[1, 1].set_xlabel('Data Points')
    axes[1, 1].set_ylabel('Data Points')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    return fig

def main():
    """Main example function."""
    print("Creating three blob dataset...")
    
    # Create the dataset
    X, y_true, connectivity = create_three_blobs_data(
        n_samples=150, 
        centers=3, 
        cluster_std=3.5, 
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Connectivity matrix shape: {connectivity.shape}")
    print(f"True cluster sizes: {np.bincount(y_true)}")
    
    # Initialize consensus clustering with hierarchical clustering
    print("\\nInitializing consensus clustering...")
    consensus_leiden = ConsensusClusteringLeiden(
        connectivity_matrix=connectivity.toarray(),
        parameter_range=(0.05, 1.0),  # Broader range for better exploration
        n_clusters_max=8,  # Allow up to 8 clusters for parameter estimation
        random_state=42,
        n_iter=30,  # More iterations for better consensus
        hierarchical_method='ward',  # Standard Ward linkage
        hierarchical_criterion='maxclust',  # Use max clusters criterion
        n_clusters=3,  # We know there should be 3 clusters
        verbose=True
    )
    
    # Fit the model
    print("\\nFitting consensus clustering...")
    consensus_leiden.fit(verbose=True)
    
    # Get results
    y_pred = consensus_leiden.predict()
    consensus_matrix = consensus_leiden.get_consensus_matrix()
    
    print(f"\\nResults:")
    print(f"Number of predicted clusters: {len(np.unique(y_pred))}")
    print(f"Predicted cluster sizes: {np.bincount(y_pred)}")
    print(f"True cluster sizes: {np.bincount(y_true)}")
    
    # Compute clustering accuracy metrics
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Normalized Mutual Information: {nmi:.3f}")
    
    # Compute and display stability metrics
    stability_metrics = consensus_leiden.compute_stability_metrics()
    print(f"\\nStability metrics:")
    for metric, value in stability_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test new stability-aware features
    print("\\nTesting stability-aware features...")
    try:
        # Get stability information
        variance_matrix = consensus_leiden.get_consensus_variance()
        confidence_matrix = consensus_leiden.get_consensus_confidence() 
        reliability_scores = consensus_leiden.get_reliability_scores()
        
        print(f"Consensus variance range: {variance_matrix.min():.4f} - {variance_matrix.max():.4f}")
        print(f"Consensus confidence range: {confidence_matrix.min():.4f} - {confidence_matrix.max():.4f}")
        print(f"Reliability scores range: {reliability_scores.min():.4f} - {reliability_scores.max():.4f}")
        
        # Compute enhanced stability metrics
        enhanced_metrics = consensus_leiden.compute_cluster_stability_metrics()
        print(f"\\nEnhanced stability metrics:")
        for metric, value in enhanced_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
                
    except Exception as e:
        print(f"Could not access stability features: {e}")
    
    # Test different hierarchical methods
    print("\\nTesting different hierarchical clustering methods...")
    methods_to_test = ['ward', 'confidence_weighted', 'stability_aware']
    
    for method in methods_to_test:
        print(f"\\n--- Testing {method} method ---")
        try:
            consensus_test = ConsensusClusteringLeiden(
                connectivity_matrix=connectivity.toarray(),
                parameter_range=(0.05, 1.0),
                n_clusters_max=8,
                random_state=42,
                n_iter=15,  # Fewer iterations for quick test
                hierarchical_method=method,
                n_clusters=3,
                verbose=False
            )
            
            consensus_test.fit(verbose=False)
            y_pred_test = consensus_test.predict()
            ari_test = adjusted_rand_score(y_true, y_pred_test)
            
            print(f"  Method: {method}")
            print(f"  Number of clusters: {len(np.unique(y_pred_test))}")
            print(f"  Cluster sizes: {np.bincount(y_pred_test)}")
            print(f"  ARI with true clusters: {ari_test:.3f}")
            
            # Get method-specific metrics
            method_metrics = consensus_test.compute_cluster_stability_metrics()
            if 'stability_score' in method_metrics:
                print(f"  Stability score: {method_metrics['stability_score']:.3f}")
            if 'consensus_silhouette' in method_metrics:
                print(f"  Consensus silhouette: {method_metrics['consensus_silhouette']:.3f}")
                
        except Exception as e:
            print(f"  Error with {method}: {e}")
    
    # Plot results
    print("\\nPlotting results...")
    fig = plot_results(X, y_true, y_pred, consensus_matrix, "Consensus Clustering: ")
    plt.savefig('three_blobs_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot consensus matrix with hierarchical ordering
    print("Plotting consensus matrix with hierarchical ordering...")
    fig, ax = consensus_leiden.plot_consensus_matrix(
        figsize=(8, 6), 
        use_hierarchical_order=True
    )
    plt.savefig('three_blobs_consensus_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot dendrogram
    print("Plotting hierarchical clustering dendrogram...")
    try:
        fig, ax = consensus_leiden.plot_dendrogram(figsize=(12, 8), show_reliability=True)
        plt.savefig('three_blobs_dendrogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot reliability heatmap
        print("Plotting reliability heatmap...")
        fig, axes = consensus_leiden.plot_reliability_heatmap(figsize=(12, 10))
        plt.savefig('three_blobs_reliability_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Show different cluster levels
        print("\\nExploring different cluster levels:")
        for n_clust in [2, 3, 4, 5]:
            labels_at_level = consensus_leiden.get_clusters_at_level(n_clust)
            ari_level = adjusted_rand_score(y_true, labels_at_level)
            print(f"  {n_clust} clusters: sizes = {np.bincount(labels_at_level)}, ARI = {ari_level:.3f}")
            
    except Exception as e:
        print(f"Could not plot dendrogram or explore cluster levels: {e}")
    
    # Try automatic cluster number determination
    print("\\nTrying automatic cluster number determination...")
    consensus_leiden_auto = ConsensusClusteringLeiden(
        connectivity_matrix=connectivity.toarray(),
        parameter_range=(0.05, 1.0),
        n_clusters_max=8,
        random_state=42,
        n_iter=20,
        hierarchical_method='ward',
        n_clusters=None,  # Let it determine automatically
        verbose=False
    )
    
    consensus_leiden_auto.fit(verbose=True)
    y_pred_auto = consensus_leiden_auto.predict()
    ari_auto = adjusted_rand_score(y_true, y_pred_auto)
    
    print(f"\\nAutomatic cluster determination:")
    print(f"Number of clusters found: {len(np.unique(y_pred_auto))}")
    print(f"Cluster sizes: {np.bincount(y_pred_auto)}")
    print(f"ARI with true clusters: {ari_auto:.3f}")
    
    # Try stability-based parameter selection
    print("\\nTrying stability-based parameter selection...")
    consensus_leiden_stable = ConsensusClusteringLeiden(
        connectivity_matrix=connectivity.toarray(),
        parameter_range=(0.05, 1.0),
        n_clusters_max=8,
        random_state=42,
        n_iter=15,
        hierarchical_method='complete',  # Try different method
        n_clusters=3,
        verbose=False
    )
    
    consensus_leiden_stable.fit_with_stability_selection(
        stability_threshold=0.7,
        verbose=True
    )
    
    y_pred_stable = consensus_leiden_stable.predict()
    ari_stable = adjusted_rand_score(y_true, y_pred_stable)
    
    print(f"\\nStability-selected results:")
    print(f"Number of clusters: {len(np.unique(y_pred_stable))}")
    print(f"Cluster sizes: {np.bincount(y_pred_stable)}")
    print(f"ARI with true clusters: {ari_stable:.3f}")
    
    # Get parameter stability scores
    stability_scores = consensus_leiden.get_parameter_stability_scores()
    print(f"\\nParameter stability scores:")
    for param, score in sorted(stability_scores.items()):
        print(f"  Parameter {param:.4f}: {score:.4f}")
    
    print("\\nThree blobs example completed successfully!")
    print("Generated files:")
    print("  - three_blobs_results.png: Overview of clustering results")
    print("  - three_blobs_consensus_matrix.png: Consensus matrix with hierarchical ordering")
    print("  - three_blobs_dendrogram.png: Hierarchical clustering dendrogram with reliability")
    print("  - three_blobs_reliability_heatmap.png: Comprehensive reliability analysis")
    print("\\nKey improvements implemented:")
    print("  ✓ Stability-aware consensus matrix computation")
    print("  ✓ Confidence and variance tracking")
    print("  ✓ Alternative hierarchical clustering methods")
    print("  ✓ Reliability-based cluster evaluation")
    print("  ✓ Enhanced visualization with stability information")

if __name__ == "__main__":
    main()