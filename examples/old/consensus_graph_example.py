"""
Example demonstrating consensus graph construction using bootstrapping methods.

This example shows how to build robust graphs from data using various
bootstrapping and subsampling techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from treeclust.utils import consensus_graph
from treeclust import Leiden

def plot_consensus_comparison(X, consensus_adj, title="Consensus Graph"):
    """Plot the data and consensus graph structure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot data points
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
    ax1.set_title(f"{title} - Data Points")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    
    # Plot consensus adjacency matrix
    im = ax2.imshow(consensus_adj, cmap='viridis', interpolation='nearest')
    ax2.set_title(f"{title} - Consensus Matrix")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Sample Index")
    plt.colorbar(im, ax=ax2, label='Consensus Weight')
    
    plt.tight_layout()
    return fig

def plot_hierarchical_consensus(X, consensus_adj, y_true, title="Hierarchical Consensus Analysis"):
    """Plot hierarchical clustering analysis of consensus matrix."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Convert consensus matrix to distance matrix
    # Distance = 1 - consensus (higher consensus = lower distance)
    consensus_distance = 1 - consensus_adj
    np.fill_diagonal(consensus_distance, 0)
    
    # Perform hierarchical clustering
    # Convert to condensed distance matrix for linkage
    n_samples = consensus_distance.shape[0]
    condensed_distances = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            condensed_distances.append(consensus_distance[i, j])
    
    # Different linkage methods
    linkage_methods = ['ward', 'complete', 'average']
    hierarchical_clusters = {}
    
    for idx, method in enumerate(linkage_methods):
        try:
            # Perform hierarchical clustering
            if method == 'ward':
                # Ward requires squared Euclidean distances
                Z = linkage(condensed_distances, method='ward')
            else:
                Z = linkage(condensed_distances, method=method)
            
            # Plot dendrogram
            axes[0, idx].set_title(f'Dendrogram - {method.capitalize()} Linkage')
            dendrogram(Z, ax=axes[0, idx], truncate_mode='level', p=10)
            axes[0, idx].set_xlabel('Sample Index')
            axes[0, idx].set_ylabel('Distance')
            
            # Get clusters (assume we want same number as true clusters)
            n_true_clusters = len(set(y_true))
            hierarchical_labels = fcluster(Z, n_true_clusters, criterion='maxclust')
            hierarchical_clusters[method] = hierarchical_labels
            
            # Plot clustering results
            axes[1, idx].scatter(X[:, 0], X[:, 1], c=hierarchical_labels, cmap='tab10', alpha=0.7)
            axes[1, idx].set_title(f'Hierarchical Clusters - {method.capitalize()}\n'
                                 f'({len(set(hierarchical_labels))} clusters)')
            axes[1, idx].set_xlabel('Feature 1')
            axes[1, idx].set_ylabel('Feature 2')
            
        except Exception as e:
            axes[0, idx].text(0.5, 0.5, f'Error with {method}: {str(e)}', 
                            transform=axes[0, idx].transAxes, ha='center', va='center')
            axes[1, idx].text(0.5, 0.5, f'Error with {method}: {str(e)}', 
                            transform=axes[1, idx].transAxes, ha='center', va='center')
    
    plt.suptitle(f'{title}\nHierarchical Clustering on Consensus Matrix', fontsize=16)
    plt.tight_layout()
    return fig, hierarchical_clusters

def plot_consensus_vs_original_comparison(X, consensus_adj, y_true, k=8, title="Consensus vs Original k-NN Comparison"):
    """Compare consensus matrix with original k-NN graph."""
    
    # Create original k-NN graph
    knn_graph = kneighbors_graph(X, n_neighbors=k, include_self=False).toarray()
    # Make symmetric (undirected)
    knn_original = (knn_graph + knn_graph.T) > 0
    knn_original = knn_original.astype(float)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Matrix visualizations
    # Original k-NN matrix
    im1 = axes[0, 0].imshow(knn_original, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('Original k-NN Graph\n(Binary Adjacency)')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Sample Index')
    plt.colorbar(im1, ax=axes[0, 0], label='Connection (0/1)')
    
    # Consensus matrix
    im2 = axes[0, 1].imshow(consensus_adj, cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title('Consensus Matrix\n(Weighted by Bootstrap Frequency)')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Sample Index')
    plt.colorbar(im2, ax=axes[0, 1], label='Consensus Weight')
    
    # Difference matrix (consensus - original)
    # Normalize original to [0,1] for fair comparison
    diff_matrix = consensus_adj - knn_original
    im3 = axes[0, 2].imshow(diff_matrix, cmap='RdBu', interpolation='nearest', 
                           vmin=-1, vmax=1)
    axes[0, 2].set_title('Difference Matrix\n(Consensus - Original)')
    axes[0, 2].set_xlabel('Sample Index')
    axes[0, 2].set_ylabel('Sample Index')
    plt.colorbar(im3, ax=axes[0, 2], label='Difference')
    
    # Edge weight distributions
    axes[0, 3].hist(knn_original[knn_original > 0], bins=20, alpha=0.7, label='Original k-NN', density=True)
    axes[0, 3].hist(consensus_adj[consensus_adj > 0], bins=20, alpha=0.7, label='Consensus', density=True)
    axes[0, 3].set_xlabel('Edge Weight')
    axes[0, 3].set_ylabel('Density')
    axes[0, 3].set_title('Edge Weight Distributions')
    axes[0, 3].legend()
    
    # Row 2: Ordered matrices by true clusters
    # Order samples by true cluster labels
    sorted_indices = np.argsort(y_true)
    
    # Original k-NN ordered
    knn_ordered = knn_original[np.ix_(sorted_indices, sorted_indices)]
    im4 = axes[1, 0].imshow(knn_ordered, cmap='viridis', interpolation='nearest')
    axes[1, 0].set_title('Original k-NN (Ordered by True Clusters)')
    axes[1, 0].set_xlabel('Sample Index (Ordered)')
    axes[1, 0].set_ylabel('Sample Index (Ordered)')
    plt.colorbar(im4, ax=axes[1, 0], label='Connection (0/1)')
    
    # Consensus ordered
    consensus_ordered = consensus_adj[np.ix_(sorted_indices, sorted_indices)]
    im5 = axes[1, 1].imshow(consensus_ordered, cmap='viridis', interpolation='nearest')
    axes[1, 1].set_title('Consensus Matrix (Ordered by True Clusters)')
    axes[1, 1].set_xlabel('Sample Index (Ordered)')
    axes[1, 1].set_ylabel('Sample Index (Ordered)')
    plt.colorbar(im5, ax=axes[1, 1], label='Consensus Weight')
    
    # Add cluster boundary lines to ordered plots
    unique_labels = np.unique(y_true)
    sorted_labels = y_true[sorted_indices]
    for label in unique_labels[:-1]:
        boundary = np.where(sorted_labels == label)[0][-1] + 0.5
        for ax in [axes[1, 0], axes[1, 1]]:
            ax.axhline(y=boundary, color='red', linestyle='--', alpha=0.7)
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
    
    # Statistics comparison
    axes[1, 2].axis('off')
    stats_text = f"""
Matrix Statistics:

Original k-NN:
  Edges: {np.sum(knn_original > 0)}
  Density: {np.mean(knn_original > 0):.3f}
  Mean degree: {np.mean(np.sum(knn_original > 0, axis=1)):.1f}

Consensus:
  Non-zero entries: {np.sum(consensus_adj > 0)}
  Density: {np.mean(consensus_adj > 0):.3f}
  Mean weight: {np.mean(consensus_adj[consensus_adj > 0]):.3f}
  
Edge Overlap:
  Shared edges: {np.sum((knn_original > 0) & (consensus_adj > 0))}
  Only in original: {np.sum((knn_original > 0) & (consensus_adj == 0))}
  Only in consensus: {np.sum((knn_original == 0) & (consensus_adj > 0))}
"""
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    # Clustering comparison
    try:
        # Cluster both matrices with Leiden
        leiden = Leiden(resolutions=1.0, random_states=0, partition_type='CPM')
        
        clusters_original = leiden.fit_predict(knn_original)
        n_clusters_original = len(set(clusters_original))
        
        clusters_consensus = leiden.fit_predict(consensus_adj)
        n_clusters_consensus = len(set(clusters_consensus))
        
        # Plot clustering results on data
        axes[1, 3].scatter(X[:, 0], X[:, 1], c=clusters_original, cmap='tab10', alpha=0.7, s=50, marker='o')
        axes[1, 3].scatter(X[:, 0], X[:, 1], c=clusters_consensus, cmap='tab20', alpha=0.7, s=20, marker='x')
        axes[1, 3].set_title(f'Clustering Comparison\n'
                           f'Original: {n_clusters_original} clusters (circles)\n'
                           f'Consensus: {n_clusters_consensus} clusters (x)')
        axes[1, 3].set_xlabel('Feature 1')
        axes[1, 3].set_ylabel('Feature 2')
        axes[1, 3].legend(['Original k-NN', 'Consensus'], loc='upper right')
        
    except Exception as e:
        axes[1, 3].text(0.5, 0.5, f'Clustering error: {str(e)}', 
                       transform=axes[1, 3].transAxes, ha='center', va='center')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Return matrices for further analysis
    return fig, knn_original, consensus_adj, diff_matrix

def main():
    # Generate test datasets
    print("Generating test datasets...")
    
    # Dataset 1: Well-separated blobs
    X_blobs, y_blobs = make_blobs(n_samples=100, centers=3, cluster_std=1.0, 
                                 center_box=(-5, 5), random_state=42)
    
    # Dataset 2: Moons (non-linear structure)
    X_moons, y_moons = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    datasets = [
        (X_blobs, y_blobs, "Blobs Dataset"),
        (X_moons, y_moons, "Moons Dataset")
    ]
    
    for X, y_true, dataset_name in datasets:
        print(f"\n{dataset_name}")
        print("=" * 50)
        
        # Method 1: Bootstrap consensus with k-NN
        print("Building consensus graph with bootstrap sampling...")
        consensus_result_knn = consensus_graph(
            X, 
            method='bootstrap_knn',
            n_iterations=50,
            sample_fraction=0.8,
            k=8,
            consensus_threshold=0.2,
            random_state=42
        )
        consensus_knn = consensus_result_knn['consensus_matrix']
        
        # Method 2: Feature subsampling consensus
        if X.shape[1] > 2:  # Only for high-dimensional data
            print("Building consensus graph with feature subsampling...")
            consensus_result_features = consensus_graph(
                X,
                method='feature_knn',
                n_iterations=50,
                feature_fraction=0.8,
                k=8,
                consensus_threshold=0.2,
                random_state=42
            )
            consensus_features = consensus_result_features['consensus_matrix']
        else:
            consensus_features = None
        
        # Analyze consensus graphs
        print(f"\nConsensus Graph Statistics for {dataset_name}:")
        print(f"k-NN Bootstrap - Edge density: {np.mean(consensus_knn > 0):.3f}, "
              f"Mean weight: {np.mean(consensus_knn[consensus_knn > 0]):.3f}")
        
        if consensus_features is not None:
            print(f"Feature Subsampling - Edge density: {np.mean(consensus_features > 0):.3f}, "
                  f"Mean weight: {np.mean(consensus_features[consensus_features > 0]):.3f}")
                
        # Test clustering on consensus graphs
        print(f"\nClustering Results for {dataset_name}:")
        
        # Cluster with Leiden on k-NN consensus
        leiden = Leiden(resolutions=0.1, random_states=0, partition_type='CPM')
        clusters_knn = leiden.fit_predict(consensus_knn)
        n_clusters_knn = len(set(clusters_knn))
        print(f"k-NN Consensus: {n_clusters_knn} clusters found")
                
        # Compare with true clusters
        print(f"True number of clusters: {len(set(y_true))}")
        
        # Plot results (if running interactively)
        try:
            # Basic consensus comparison
            fig1 = plot_consensus_comparison(X, consensus_knn, f"{dataset_name} - k-NN Bootstrap")
            
            # Hierarchical analysis
            fig2, hierarchical_clusters = plot_hierarchical_consensus(
                X, consensus_knn, y_true, f"{dataset_name} - Hierarchical Analysis"
            )
            
            # NEW: Consensus vs Original k-NN comparison
            fig3, knn_original, consensus_matrix, diff_matrix = plot_consensus_vs_original_comparison(
                X, consensus_knn, y_true, k=8, title=f"{dataset_name} - Consensus vs Original k-NN"
            )
            
            if consensus_features is not None:
                fig4 = plot_consensus_comparison(X, consensus_features, f"{dataset_name} - Feature Subsampling")
            
            # Show clustering results comparison
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Row 1: Original clusters and Leiden clustering
            axes[0, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', alpha=0.7)
            axes[0, 0].set_title(f"True Clusters ({len(set(y_true))})")
            
            axes[0, 1].scatter(X[:, 0], X[:, 1], c=clusters_knn, cmap='tab10', alpha=0.7)
            axes[0, 1].set_title(f"Leiden on Consensus ({n_clusters_knn} clusters)")
            
            # Show consensus matrix structure
            im = axes[0, 2].imshow(consensus_knn, cmap='viridis', interpolation='nearest')
            axes[0, 2].set_title("Consensus Matrix")
            plt.colorbar(im, ax=axes[0, 2], label='Consensus Weight')
            
            # Row 2: Hierarchical clustering results
            methods = ['ward', 'complete', 'average']
            for idx, method in enumerate(methods):
                if method in hierarchical_clusters:
                    labels = hierarchical_clusters[method]
                    axes[1, idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.7)
                    axes[1, idx].set_title(f"Hierarchical - {method.capitalize()}\n"
                                         f"({len(set(labels))} clusters)")
                else:
                    axes[1, idx].text(0.5, 0.5, f'No {method} results', 
                                    transform=axes[1, idx].transAxes, ha='center', va='center')
                                    
            for ax in axes.flat:
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
            
            plt.suptitle(f'{dataset_name} - Clustering Methods Comparison', fontsize=16)
            plt.tight_layout()
            
            # Print comparison statistics
            print(f"\nConsensus vs Original k-NN Comparison for {dataset_name}:")
            print(f"Original k-NN edges: {np.sum(knn_original > 0)}")
            print(f"Consensus non-zero entries: {np.sum(consensus_matrix > 0)}")
            print(f"Shared edges: {np.sum((knn_original > 0) & (consensus_matrix > 0))}")
            print(f"Edges only in original: {np.sum((knn_original > 0) & (consensus_matrix == 0))}")
            print(f"Edges only in consensus: {np.sum((knn_original == 0) & (consensus_matrix > 0))}")
            
            # Calculate edge stability
            shared_edges = np.sum((knn_original > 0) & (consensus_matrix > 0))
            total_original_edges = np.sum(knn_original > 0)
            edge_stability = shared_edges / total_original_edges if total_original_edges > 0 else 0
            print(f"Edge stability (shared/original): {edge_stability:.3f}")
            
            # Print hierarchical clustering statistics
            print(f"\nHierarchical Clustering Results for {dataset_name}:")
            for method, labels in hierarchical_clusters.items():
                n_clusters = len(set(labels))
                print(f"{method.capitalize()} linkage: {n_clusters} clusters")
                
                # Calculate cluster size distribution
                unique_labels, counts = np.unique(labels, return_counts=True)
                size_dist = dict(zip(unique_labels, counts))
                print(f"  Cluster sizes: {size_dist}")
            
            plt.show()
            
        except Exception as e:
            print(f"Plotting not available: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()