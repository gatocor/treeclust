"""
Example demonstrating the CoassociationDistanceMatrix class with hierarchical clustering.

This example shows how to use the CoassociationDistanceMatrix class with its built-in
plotting methods for ensemble clustering and hierarchical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

# Import TreeClust classes
from treeclust import Leiden
from treeclust import ConsensusNearestNeighbors
from treeclust import CoassociationDistanceMatrix
from treeclust.bootstrapping.data_bootstrapper import DataBootstrapper

import igraph as ig

# Generate synthetic data for testing
np.random.seed(42)
X, true_labels = make_blobs(n_samples=2000, centers=4, n_features=2, 
                           cluster_std=2.5, random_state=42)

leiden = Leiden(
    resolution=0.001,
    partition_type='CPM',
    # connectivity=ConsensusNearestNeighbors().fit_transform,
    flavor='auto'
)

coassotiation = CoassociationDistanceMatrix(
    clustering_classes=leiden,
    n_splits=10
    # max_runs=1000
)

# Create and show the original data
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X[:, 0], y=X[:, 1],
    hue=true_labels,
    palette='tab10',
    s=50,
    edgecolor='k',
)
plt.title('Original Data with True Clusters')
plt.legend(title='True Cluster')

# Fit the coassociation matrix
print("Fitting coassociation matrix...")

# Debug the bootstrapping and clustering process
print("\n=== BOOTSTRAPPING DEBUG ===")
from treeclust.bootstrapping.data_bootstrapper import DataBootstrapper

# Test bootstrapping function
bootstrap_func = DataBootstrapper().sample
bootstrap_result = bootstrap_func(X)
X_bootstrap = bootstrap_result[0]
train_indices = bootstrap_result[2]

print(f"Original data shape: {X.shape}")
print(f"Bootstrap sample shape: {X_bootstrap.shape}")
print(f"Bootstrap indices: {len(train_indices)} samples")
print(f"Bootstrap index range: {min(train_indices)} to {max(train_indices)}")

# Test what Leiden produces on bootstrap sample
test_leiden_bootstrap = Leiden(resolution=0.01, partition_type='CPM', flavor='auto')
bootstrap_labels = test_leiden_bootstrap.fit_predict(X_bootstrap)
print(f"Bootstrap Leiden produces {len(np.unique(bootstrap_labels))} clusters: {np.unique(bootstrap_labels)}")
print(f"Bootstrap cluster sizes: {np.bincount(bootstrap_labels)}")

# Test multiple bootstrap runs to see stability
print(f"\nTesting multiple bootstrap runs:")
for run in range(5):
    bootstrap_result = bootstrap_func(X)
    X_boot = bootstrap_result[0]
    train_idx = bootstrap_result[2]
    boot_labels = test_leiden_bootstrap.fit_predict(X_boot)
    print(f"Run {run+1}: {len(train_idx)} samples, {len(np.unique(boot_labels))} clusters, sizes: {np.bincount(boot_labels)}")

print("=== END BOOTSTRAPPING DEBUG ===\n")

normalized_coassoc = coassotiation.fit(X).get_coassociation_matrix().toarray()
print(f"Coassociation matrix shape: {normalized_coassoc.shape}")

# Test what a single run of Leiden produces
print("\n=== LEIDEN CLUSTERING TEST ===")
test_leiden = Leiden(resolution=0.01, partition_type='CPM', flavor='auto')
test_labels = test_leiden.fit_predict(X)
print(f"Single Leiden run produces {len(np.unique(test_labels))} clusters: {np.unique(test_labels)}")

# Check how well it matches true labels
for cluster_id in np.unique(test_labels):
    mask = test_labels == cluster_id
    true_in_cluster = true_labels[mask]
    print(f"Leiden cluster {cluster_id} ({np.sum(mask)} samples): true labels {dict(zip(*np.unique(true_in_cluster, return_counts=True)))}")

# Check coassociation matrix properties
print(f"\nCoassociation matrix stats:")
print(f"Min: {normalized_coassoc.min():.3f}, Max: {normalized_coassoc.max():.3f}, Mean: {normalized_coassoc.mean():.3f}")
print(f"Diagonal (should be 1.0): {normalized_coassoc.diagonal()[:10]}")  

# Check if clusters are well-separated in coassociation matrix
print(f"Off-diagonal values (should be low for well-separated clusters):")
off_diag_values = normalized_coassoc[~np.eye(normalized_coassoc.shape[0], dtype=bool)]
print(f"Off-diagonal: min={off_diag_values.min():.3f}, max={off_diag_values.max():.3f}, mean={off_diag_values.mean():.3f}")
print("=== END LEIDEN TEST ===\n")

# Perform hierarchical clustering
print("Performing hierarchical clustering...")

# Let's debug the hierarchical clustering step by step
# Import functions we need (scipy should already be imported in coassotiation.py)

# Get the distance matrix that will be used for hierarchical clustering
distance_matrix = coassotiation.get_distance_matrix().toarray()
print(f"\n=== DISTANCE MATRIX DEBUG ===")
print(f"Distance matrix shape: {distance_matrix.shape}")
print(f"Distance matrix stats: min={distance_matrix.min():.3f}, max={distance_matrix.max():.3f}, mean={distance_matrix.mean():.3f}")
print(f"Diagonal (should be 0.0): {distance_matrix.diagonal()[:10]}")

# Check if distance matrix is symmetric
is_symmetric = np.allclose(distance_matrix, distance_matrix.T)
print(f"Distance matrix is symmetric: {is_symmetric}")

# Let's check what the coassociation matrix looks like for the first few samples
print(f"Coassociation matrix (first 10x10):")
print(normalized_coassoc[:10, :10])
print(f"Distance matrix (first 10x10):")
print(distance_matrix[:10, :10])

print("=== END DISTANCE MATRIX DEBUG ===\n")

hierarchical_result = coassotiation.hierarchical_clustering(
    linkage_method='complete',
    max_hierarchy_levels=5,
    min_cluster_size=10
)

# Debug the hierarchical clustering results
print("\n=== HIERARCHICAL CLUSTERING DEBUG ===")
print(f"Hierarchical result keys: {list(hierarchical_result.keys())}")

cluster_assignments = hierarchical_result.get('cluster_assignments', {})
print(f"Found {len(cluster_assignments)} hierarchy levels: {sorted(cluster_assignments.keys())}")

for level in sorted(cluster_assignments.keys()):
    assignments = cluster_assignments[level]
    unique_clusters = np.unique(assignments)
    print(f"Level {level}: {len(unique_clusters)} clusters")
    for cluster_id in unique_clusters:
        count = np.sum(assignments == cluster_id)
        print(f"  Cluster {cluster_id}: {count} samples")

# Debug cluster stability
cluster_stability = hierarchical_result.get('cluster_stability', {})
print(f"\nCluster stability keys: {sorted(cluster_stability.keys())}")
for level in sorted(cluster_stability.keys()):
    stab = cluster_stability[level]
    print(f"Level {level} stability: {stab}")

# Get the hierarchy graph
hierarchy_graph = hierarchical_result['hierarchy_graph']
print(f"\nHierarchy graph type: {type(hierarchy_graph)}")
if hasattr(hierarchy_graph, 'vs'):
    print(f"Graph has {len(hierarchy_graph.vs)} vertices and {len(hierarchy_graph.es)} edges")
    print(f"Vertex levels: {sorted(set(hierarchy_graph.vs['level']))}")
    for level in sorted(set(hierarchy_graph.vs['level'])):
        vertices_at_level = [v for v in hierarchy_graph.vs if v['level'] == level]
        print(f"  Level {level}: {len(vertices_at_level)} clusters")
elif isinstance(hierarchy_graph, dict):
    print(f"Dict keys: {list(hierarchy_graph.keys())}")
    if 'nodes' in hierarchy_graph:
        print(f"Number of nodes: {len(hierarchy_graph['nodes'])}")
        print(f"Number of edges: {len(hierarchy_graph['edges'])}")
print("=== END DEBUG ===\n")

for level in coassotiation.get_hierarchy_levels():
    clusters = coassotiation.get_level_clusters(level)
    soft_labels = coassotiation.get_cluster_assignment_probabilities({cluster:i for i,cluster in enumerate(clusters)})
    print(soft_labels)
    labels = np.argmax(soft_labels, axis=1)
    w = soft_labels.max(axis=1)
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X[:, 0], y=X[:, 1],
        hue=labels,
        palette='tab10',
        size_norm=(0.0, 1.0),
        sizes=(20,100),
        size=w,
    )
    plt.title(f'Hierarchical Clustering Level {level} with {len(np.unique(labels))} Clusters')
    plt.legend(title='Cluster')
    # plt.show()

# Use the new class methods for visualization
print("Creating aligned visualization with graph...")
fig = coassotiation.plot_combined_analysis_aligned(
    true_labels=true_labels,
    figsize=(16, 8)
)

print("Creating visualization with stacked bars...")
fig = coassotiation.plot_combined_analysis_stacked(
    true_labels=true_labels,
    figsize=(16, 8)
)
plt.show()