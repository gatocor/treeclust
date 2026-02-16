from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from treeclust.neighbors import NearestNeighbors, ConsensusNearestNeighbors, MutualNearestNeighbors, ConsensusMutualNearestNeighbors
from treeclust.pipelines import PipelineBootstrapper
from treeclust.decomposition import PCA
from treeclust.clustering import MultiresolutionLeiden
from treeclust.plotting import plot_graph, plot_multiresolution_hierarchy, plot_multiresolution_graph, plot_cluster_expression_dotplot
from treeclust.annotation import annotation_hierarchical, propagate_annotation
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

X, y = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=0.60, random_state=0)
# Add random noise in N additional dimensions
N = 0  # Number of additional noisy dimensions
if N > 0:
    rng = np.random.RandomState(42)  # Create random state for reproducible noise
    X = np.hstack([X, rng.normal(size=(X.shape[0], N))])
else:
    X = X

# Create a KNN graph
pipeline = PipelineBootstrapper(
    steps=[
        # ('pca', PCA(n_components=2))
        ],
    # feature_splitter=ShuffleSplit(random_state=42),
    observation_splitter=ShuffleSplit(random_state=42)
    )
knn = ConsensusMutualNearestNeighbors(
        n_neighbors=15, 
        keep_bootstrap_matrices=True, 
        n_splits=10, 
        pipeline_bootstrapper=pipeline
    )
matrix = knn.fit_transform(X)

clustering = MultiresolutionLeiden(random_state=42)
clustering.fit(matrix)

# metrics = clustering.get_resolution_range_summary()
# fig, ax = plt.subplots(1,3,figsize=(10,5))
# ax[0].plot(metrics['resolution_values'], metrics['n_clusters'], marker='o')
# ax[1].plot(metrics['resolution_values'], metrics['consistencies'], marker='o')
# ax[2].plot(metrics['n_clusters'], metrics['consistencies'], marker='o')

print(clustering.get_observed_clusters())
print(clustering.get_observed_confidence())
print(clustering.get_cluster_metric())

fig, ax = plt.subplots(1,1,figsize=(10,5))
plot_multiresolution_hierarchy(
        clustering.get_observed_clusters(),
        size_dict=clustering.get_cluster_metric(metric="stability"),
        layout_algorithm="organic",
        ax=ax
    )

fig = plot_multiresolution_graph(
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