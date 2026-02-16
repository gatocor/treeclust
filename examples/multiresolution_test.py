import scanpy as scfrom sklearn.datasets import make_blobs

import treeclust.scanpy as tcscimport matplotlib.pyplot as plt

import numpy as npimport pandas as pd

from sklearn.model_selection import ShuffleSplit

# Load example data

adata = sc.datasets.pbmc3k_processed()from treeclust.neighbors import NearestNeighbors, ConsensusNearestNeighbors, MutualNearestNeighbors, ConsensusMutualNearestNeighbors

print(f"Data shape: {adata.shape}")from treeclust.pipelines import PipelineBootstrapper

from treeclust.decomposition import PCA

# First compute neighbors (required for clustering)from treeclust.clustering import MultiresolutionLeiden

print("\n=== Computing neighbors ===")from treeclust.plotting import plot_graph, plot_multiresolution_hierarchy, plot_multiresolution_graph, plot_cluster_expression_dotplot

tcsc.pp.neighbors(adata, n_neighbors=15, key_added='treeclust_neighbors')from treeclust.annotation import annotation_hierarchical, propagate_annotation

import numpy as np

# Test different resolution parameter formats

print("\n=== Testing multiresolution leiden with different resolution formats ===")# Set random seed for reproducibility

np.random.seed(42)

# 1. Default range (0, 1)

print("\n1. Default range (0, 1)...")X, y = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=0.60, random_state=0)

tcsc.tl.multiresolution_leiden(# Add random noise in N additional dimensions

    adata, N = 0  # Number of additional noisy dimensions

    neighbors_key='treeclust_neighbors',if N > 0:

    key_added='default_range'    rng = np.random.RandomState(42)  # Create random state for reproducible noise

)    X = np.hstack([X, rng.normal(size=(X.shape[0], N))])

print(f"Default range results: {[col for col in adata.obs.columns if 'default_range' in col]}")else:

print(f"Resolution mode: {adata.uns['default_range']['resolution_mode']}")    X = X

print(f"Resolution range: {adata.uns['default_range']['resolution_range']}")

print(f"Actual resolutions: {adata.uns['default_range']['resolutions']}")# Create a KNN graph

pipeline = PipelineBootstrapper(

# 2. Custom range    steps=[

print("\n2. Custom range (0.1, 2.0)...")        # ('pca', PCA(n_components=2))

tcsc.tl.multiresolution_leiden(        ],

    adata,     # feature_splitter=ShuffleSplit(random_state=42),

    resolutions=(0.1, 2.0),    observation_splitter=ShuffleSplit(random_state=42)

    neighbors_key='treeclust_neighbors',    )

    key_added='custom_range'knn = ConsensusMutualNearestNeighbors(

)        n_neighbors=15, 

print(f"Custom range results: {[col for col in adata.obs.columns if 'custom_range' in col]}")        keep_bootstrap_matrices=True, 

print(f"Resolution mode: {adata.uns['custom_range']['resolution_mode']}")        n_splits=10, 

print(f"Resolution range: {adata.uns['custom_range']['resolution_range']}")        pipeline_bootstrapper=pipeline

print(f"Actual resolutions: {adata.uns['custom_range']['resolutions']}")    )

matrix = knn.fit_transform(X)

# 3. Specific values (list)

print("\n3. Specific values [0.2, 0.5, 1.0]...")clustering = MultiresolutionLeiden(random_state=42)

tcsc.tl.multiresolution_leiden(clustering.fit(matrix)

    adata, 

    resolutions=[0.2, 0.5, 1.0],# metrics = clustering.get_resolution_range_summary()

    neighbors_key='treeclust_neighbors',# fig, ax = plt.subplots(1,3,figsize=(10,5))

    key_added='specific_values'# ax[0].plot(metrics['resolution_values'], metrics['n_clusters'], marker='o')

)# ax[1].plot(metrics['resolution_values'], metrics['consistencies'], marker='o')

print(f"Specific values results: {[col for col in adata.obs.columns if 'specific_values' in col]}")# ax[2].plot(metrics['n_clusters'], metrics['consistencies'], marker='o')

print(f"Resolution mode: {adata.uns['specific_values']['resolution_mode']}")

print(f"Resolution range: {adata.uns['specific_values']['resolution_range']}")print(clustering.get_observed_clusters())

print(f"Actual resolutions: {adata.uns['specific_values']['resolutions']}")print(clustering.get_observed_confidence())

print(clustering.get_cluster_metric())

# Check cluster counts for each approach

print("\n=== Cluster counts comparison ===")fig, ax = plt.subplots(1,1,figsize=(10,5))

for approach in ['default_range', 'custom_range', 'specific_values']:plot_multiresolution_hierarchy(

    cols = [col for col in adata.obs.columns if approach in col]        clustering.get_observed_clusters(),

    for col in sorted(cols):        size_dict=clustering.get_cluster_metric(metric="stability"),

        n_clusters = adata.obs[col].nunique()        layout_algorithm="organic",

        res = col.split('_')[-1]        ax=ax

        print(f"{approach} resolution {res}: {n_clusters} clusters")    )



print("\n=== Success! Multiresolution with range/values works ===")fig = plot_multiresolution_graph(
    edge_width=matrix,
    node_color=clustering.labels_,
    node_size=clustering.cell_confidence_,
    node_size_norm=(0.,1),
    node_sizes=(0,100),
    node_pos=X[:,:2],
    figsize=(15,10),
    suptitle="Multiresolution Clustering Results"
)

# Plot the results

annotation = {
    (3, 2): "A",
    (6, 7): "B",
    (6, 1): "C",
    (6, 6): "C",
    (7, 1): "D"
}

labels_proba = annotation_hierarchical(clustering.labels_, annotation, weights_dict=clustering.cell_confidence_)
labels_proba_propagated = propagate_annotation(matrix, labels_proba)
labels = labels_proba.idxmax(axis=1).values
labels_propagated = labels_proba_propagated.idxmax(axis=1).values
print(labels_proba)

fig, ax = plt.subplots(1,2,figsize=(8, 8))
fig_graph = plot_graph(
        edge_width=matrix, 
        node_pos=X[:,:2], 
        node_color=labels,
        node_size=labels_proba.max(axis=1).values,
        show_labels=True,
        ax=ax[0]
    )
ax[0].axis('off')
fig_graph = plot_graph(
        edge_width=knn.matrix_, 
        node_pos=X[:,:2], 
        node_color=labels_propagated,
        node_size=labels_proba_propagated.max(axis=1).values,
        show_labels=True,
        ax=ax[1]
    )
ax[1].axis('off')
fig.tight_layout()

fig, ax = plt.subplots(figsize=(8, 8))
plot_cluster_expression_dotplot(
    expression_data=pd.DataFrame(X),
    clustering_results=clustering.labels_,
    ax=ax,
    figsize=(8,8),
    # suptitle="Cluster Expression Dotplot"
)
fig.tight_layout()

plt.show()