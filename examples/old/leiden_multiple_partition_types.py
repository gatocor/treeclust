"""
Example demonstrating the Leiden class with multiple partition types.

This example shows how to use multiple partition types (CPM, modularity, etc.)
with the same resolution and random state parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from treeclust import Leiden

def plot_partition_type_comparison(X, y_true, partitions, leiden_obj, title="Partition Type Comparison"):
    """Plot clustering results for different partition types."""
    
    partition_types = leiden_obj.partition_types
    resolutions = leiden_obj.resolutions
    random_states = leiden_obj.random_states
    
    # Create subplot grid: partition types × (resolutions or random states)
    # For simplicity, let's compare partition types for first resolution and first random state
    first_resolution = resolutions[0]
    first_random_state = random_states[0]
    
    n_types = len(partition_types)
    fig, axes = plt.subplots(1, n_types + 1, figsize=(4 * (n_types + 1), 4))
    
    if n_types == 0:
        axes = [axes]
    
    # Plot true clusters
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', alpha=0.7, s=50)
    axes[0].set_title(f'True Clusters\n({len(set(y_true))} clusters)')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Plot each partition type
    for i, partition_type in enumerate(partition_types):
        ax = axes[i + 1]
        key = f"partition_{partition_type}_resolution_{first_resolution}_seed_{first_random_state}"
        
        if key in partitions:
            labels = partitions[key]
            n_clusters = len(set(labels))
            
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.7, s=50)
            ax.set_title(f'{partition_type}\n({n_clusters} clusters)')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
    
    plt.suptitle(f'{title}\nResolution={first_resolution:.3f}, Random State={first_random_state}', fontsize=14)
    plt.tight_layout()
    return fig

# Generate sample data
X, y_true = make_blobs(n_samples=150, centers=4, cluster_std=1.5, 
                      center_box=(-10, 10), random_state=42)

# Create adjacency matrix using k-nearest neighbors
adjacency_matrix = kneighbors_graph(X, n_neighbors=15, include_self=False).toarray()

# Make it symmetric (undirected)
adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) > 0
adjacency_matrix = adjacency_matrix.astype(int)

# Initialize Leiden clustering with multiple partition types
leiden = Leiden(
    resolutions=[0.5, 1.0, 2.0],
    random_states=[0, 1, 2],
    partition_type=['CPM', 'modularity', 'surprise']  # Multiple partition types
)

print(f"Leiden configuration:")
print(f"  Resolutions: {leiden.resolutions}")
print(f"  Random states: {leiden.random_states}")
print(f"  Partition types: {leiden.partition_types}")

# Fit the model and get all partitions
print(f"\nFitting model...")
partitions = leiden.fit_predict(adjacency_matrix)

print(f"Generated {len(partitions)} partitions")
print(f"Example partition keys:")
for key in list(partitions.keys())[:5]:
    print(f"  {key}")

# Get number of clusters for each partition
n_clusters = leiden.get_n_clusters()
print(f"\nNumber of clusters per partition type (first resolution, first random state):")
for partition_type in leiden.partition_types:
    first_resolution = leiden.resolutions[0]
    first_random_state = leiden.random_states[0]
    key = f"partition_{partition_type}_resolution_{first_resolution}_seed_{first_random_state}"
    if key in n_clusters:
        print(f"  {partition_type}: {n_clusters[key]} clusters")

# Compare specific partitions
print(f"\nComparing partition types (resolution={leiden.resolutions[0]:.3f}, random_state={leiden.random_states[0]}):")
for partition_type in leiden.partition_types:
    partition = leiden.get_partition(
        resolution=leiden.resolutions[0], 
        random_state=leiden.random_states[0],
        partition_type=partition_type
    )
    n_clust = len(set(partition))
    print(f"  {partition_type}: {n_clust} clusters")

print(f"\nLeiden object: {leiden}")

# Plotting section
print("\nGenerating plots...")

try:
    # Plot comparison of different partition types
    fig = plot_partition_type_comparison(X, y_true, partitions, leiden)
    
    # Show the plot
    plt.show()
    
    print(f"\nExample completed successfully!")
    print(f"Compared {len(leiden.partition_types)} partition types with "
          f"{len(leiden.resolutions)} resolutions × {len(leiden.random_states)} random states")
    
except Exception as e:
    print(f"Plotting failed: {e}")
    import traceback
    traceback.print_exc()