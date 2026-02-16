"""
Example demonstrating the new ConsensusCluster class with Leiden partitions.

This example shows how to:
1. Generate multiple partitions using the Leiden class
2. Build consensus using the ConsensusCluster class with the theory from the paper
3. Visualize different types of coassociation matrices
4. Compare consensus results and stability metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from treeclust import Leiden, ConsensusCluster

def plot_hierarchical_clustermap(consensus_cluster, figsize=(12, 10), 
                                true_labels=None, consensus_labels=None):
    """
    Plot hierarchical clustermap with properly aligned dendrograms.
    
    Parameters:
    -----------
    consensus_cluster : ConsensusCluster
        Fitted consensus clustering object
    matrix_type : str, default='basic'
        Type of coassociation matrix to plot
    figsize : tuple, default=(12, 10)
        Figure size
    true_labels : array-like, optional
        True cluster labels for annotation
    consensus_labels : array-like, optional
        Consensus cluster labels for annotation
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Get the coassociation matrix
    coassoc_matrix = consensus_cluster.get_coassociation_matrix()
    
    # Convert to distance matrix for hierarchical clustering
    distance_matrix = 1 - coassoc_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Create figure with custom gridspec for better control
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout: dendrogram space and heatmap space
    gs = fig.add_gridspec(4, 4, hspace=0.02, wspace=0.02,
                         height_ratios=[1, 0.1, 3, 0.5], 
                         width_ratios=[1, 0.1, 3, 0.5])
    
    # Use the linkage matrix from consensus cluster if available
    if hasattr(consensus_cluster, 'linkage_matrix_') and consensus_cluster.linkage_matrix_ is not None:
        Z = consensus_cluster.linkage_matrix_
    else:
        # Fallback to computing linkage
        condensed_distances = squareform(distance_matrix)
        Z = linkage(condensed_distances, method=consensus_cluster.linkage_method)
    
    # Top dendrogram
    ax_dendro_top = fig.add_subplot(gs[0, 2])
    dendro_top = dendrogram(Z, orientation='top', ax=ax_dendro_top, 
                          color_threshold=0, above_threshold_color='black')
    ax_dendro_top.set_xticks([])
    ax_dendro_top.set_yticks([])
    ax_dendro_top.spines['top'].set_visible(False)
    ax_dendro_top.spines['right'].set_visible(False)
    ax_dendro_top.spines['bottom'].set_visible(False)
    ax_dendro_top.spines['left'].set_visible(False)
    
    # Get the order from top dendrogram
    leaf_order = dendro_top['leaves']
    
    # Left dendrogram - use the same ordering as top
    ax_dendro_left = fig.add_subplot(gs[2, 0])
    # For left dendrogram, we need to match the same sample ordering
    dendro_left = dendrogram(Z, orientation='left', ax=ax_dendro_left,
                           color_threshold=0, above_threshold_color='black')
    ax_dendro_left.set_xticks([])
    ax_dendro_left.set_yticks([])
    ax_dendro_left.spines['top'].set_visible(False)
    ax_dendro_left.spines['right'].set_visible(False)
    ax_dendro_left.spines['bottom'].set_visible(False)
    ax_dendro_left.spines['left'].set_visible(False)
    # Invert the y-axis to match the heatmap orientation (top to bottom)
    ax_dendro_left.invert_yaxis()
    
    # Reorder the matrix according to the dendrogram
    reordered_matrix = coassoc_matrix[np.ix_(leaf_order, leaf_order)]
    
    # Main heatmap
    ax_heatmap = fig.add_subplot(gs[2, 2])
    im = ax_heatmap.imshow(reordered_matrix, cmap='viridis', aspect='auto', 
                          interpolation='nearest')
    ax_heatmap.set_title(f'Coassociation Matrix\n'
                        f'Linkage: {consensus_cluster.linkage_method}')
    ax_heatmap.set_xlabel('Samples (reordered)')
    ax_heatmap.set_ylabel('Samples (reordered)')
    
    # Add colorbar
    ax_colorbar = fig.add_subplot(gs[2, 3])
    cbar = plt.colorbar(im, cax=ax_colorbar)
    cbar.set_label('Coassociation Strength')
    
    # Add annotation bars if labels provided
    if true_labels is not None or consensus_labels is not None:
        # Bottom annotation bar for true labels
        if true_labels is not None:
            ax_true_annot = fig.add_subplot(gs[3, 2])
            true_reordered = np.array(true_labels)[leaf_order]
            # Create color map for true labels
            unique_true = np.unique(true_labels)
            colors_true = plt.cm.Set1(np.linspace(0, 1, len(unique_true)))
            color_map_true = {label: colors_true[i] for i, label in enumerate(unique_true)}
            true_colors = [color_map_true[label] for label in true_reordered]
            
            ax_true_annot.imshow([true_colors], aspect='auto')
            ax_true_annot.set_xlim(ax_heatmap.get_xlim())
            ax_true_annot.set_xticks([])
            ax_true_annot.set_yticks([])
            ax_true_annot.set_ylabel('True\nLabels', rotation=0, ha='right', va='center')
            
        # Right annotation bar for consensus labels  
        if consensus_labels is not None:
            ax_consensus_annot = fig.add_subplot(gs[2, 3])
            if hasattr(ax_consensus_annot, 'clear'):
                ax_consensus_annot.clear()  # Clear the colorbar axis
            
            # Use a different subplot for consensus annotation
            ax_consensus_annot = fig.add_subplot(gs[1, 2])
            consensus_reordered = np.array(consensus_labels)[leaf_order]
            # Create color map for consensus labels
            unique_consensus = np.unique(consensus_labels)
            colors_consensus = plt.cm.Set3(np.linspace(0, 1, len(unique_consensus)))
            color_map_consensus = {label: colors_consensus[i] for i, label in enumerate(unique_consensus)}
            consensus_colors = [color_map_consensus[label] for label in consensus_reordered]
            
            ax_consensus_annot.imshow([consensus_colors], aspect='auto')
            ax_consensus_annot.set_xlim(ax_heatmap.get_xlim())
            ax_consensus_annot.set_xticks([])
            ax_consensus_annot.set_yticks([])
            ax_consensus_annot.set_ylabel('Consensus\nLabels', rotation=0, ha='right', va='center')
    
    # Add statistics text
    mean_coassoc = np.mean(reordered_matrix)
    std_coassoc = np.std(reordered_matrix)
    
    stats_text = f'Mean: {mean_coassoc:.3f}\nStd: {std_coassoc:.3f}\nSamples: {reordered_matrix.shape[0]}'
    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return fig

# Generate sample data with clear cluster structure
print("Generating sample data...")
X, y_true = make_blobs(n_samples=150, centers=4, cluster_std=3.5, 
                      center_box=(-12, 12), random_state=42)

print(f"Generated data: {X.shape[0]} samples, {len(set(y_true))} true clusters")

# Create adjacency matrix using k-nearest neighbors
print("Building adjacency matrix...")
adjacency_matrix = kneighbors_graph(X, n_neighbors=10, include_self=False).toarray()

# Make it symmetric (undirected)
adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) > 0
adjacency_matrix = adjacency_matrix.astype(int)

print(f"Adjacency matrix: {adjacency_matrix.shape}, density: {np.mean(adjacency_matrix):.3f}")

# Generate multiple partitions using Leiden clustering
print("\nGenerating Leiden partitions...")
leiden = Leiden(
    resolutions=(0,2),  # Range of resolutions
    random_states=100,  # Multiple random seeds
    partition_type=['CPM', 'modularity']  # Multiple partition types
)

# Fit and get all partitions
leiden_partitions = leiden.fit_predict(X)

print(f"Generated {len(leiden_partitions)} partitions:")
print(f"  Resolution range: {min(leiden.resolutions):.3f} - {max(leiden.resolutions):.3f}")
print(f"  Random states: {leiden.random_states}")
print(f"  Partition types: {leiden.partition_types}")

# Show sample partition statistics
print(f"\nSample partition statistics:")
sample_keys = list(leiden_partitions.keys())[:5]
for key in sample_keys:
    partition = leiden_partitions[key]
    n_clusters = len(set(partition))
    print(f"  {key}: {n_clusters} clusters")

# Build consensus using ConsensusCluster
print("\n" + "="*50)
print("BUILDING CONSENSUS")
print("="*50)

consensus_cluster = ConsensusCluster(
    linkage_method='complete'
)

consensus_cluster_weighted = ConsensusCluster(
    linkage_method='complete',
    weighting_type='weighted'
)


print("Fitting consensus model...")
# Fit consensus model with all partitions
consensus_cluster.fit(
    partitions=leiden_partitions
)

consensus_cluster_weighted.fit(
    partitions=leiden_partitions,
    reference_partition=y_true
)

fig = plt.figure(figsize=(8, 6))


fig2 = plot_hierarchical_clustermap(
    consensus_cluster, 
    figsize=(14, 12),
    true_labels=y_true,
    # consensus_labels=consensus_labels_for_plot
)

fig2 = plot_hierarchical_clustermap(
    consensus_cluster_weighted, 
    figsize=(14, 12),
    true_labels=y_true,
    # consensus_labels=consensus_labels_for_plot
)

# # Cluster stability analysis
# print("Generating cluster stability analysis...")
# fig3, stability_metrics = plot_cluster_stability_analysis(
#     consensus_cluster, 
#     y_true, 
#     consensus_labels_for_plot,
#     figsize=(16, 12)
# )
        
plt.show()