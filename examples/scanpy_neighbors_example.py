import scanpy as sc
import treeclust.scanpy as tcsc
import matplotlib.pyplot as plt
from treeclust.pipelines import PipelineBootstrapper
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA

# Load example data
adata = sc.datasets.pbmc3k_processed()
print(f"Data shape: {adata.shape}")
print(f"Original uns keys: {list(adata.uns.keys())}")

# Test different neighbor methods
print("\n=== Testing treeclust scanpy-style preprocessing ===")

bootstrapper = PipelineBootstrapper(
    steps=[
        ('pca', PCA(n_components=30))
    ],
    observation_splitter=ShuffleSplit(test_size=0.2),
    feature_splitter=ShuffleSplit(test_size=0.2)
)

# 1. Standard k-nearest neighbors (equivalent to scanpy.pp.neighbors)
print("\n1. Standard neighbors...")
tcsc.pp.neighbors(adata, n_neighbors=15, key_added='treeclust_neighbors')
print(f"After neighbors: {list(adata.uns.keys())}")
print(f"Neighbors keys: {list(adata.uns['treeclust_neighbors'].keys())}")

# 2. Consensus neighbors for robustness
print("\n2. Consensus neighbors...")
tcsc.pp.consensus_neighbors(adata, n_neighbors=15, n_splits=10, key_added='consensus_neighbors', pipeline_bootstrapper=bootstrapper)
print(f"After consensus: {list(adata.uns.keys())}")

# 3. Mutual nearest neighbors for symmetry
print("\n3. Mutual neighbors...")
tcsc.pp.mutual_neighbors(adata, n_neighbors=15, key_added='mutual_neighbors')
print(f"After mutual: {list(adata.uns.keys())}")

# 4. Combined consensus + mutual approach
print("\n4. Consensus mutual neighbors...")
tcsc.pp.consensus_mutual_neighbors(adata, n_neighbors=15, n_splits=5, key_added='consensus_mutual')
print(f"After consensus mutual: {list(adata.uns.keys())}")

# Check shapes of different matrix types
print("\n=== Matrix types and shapes ===")
for key in ['treeclust_neighbors', 'consensus_neighbors', 'mutual_neighbors', 'consensus_mutual']:
    if key in adata.uns:
        for matrix_type in ['distances', 'connectivities', 'transitions']:
            matrix = adata.uns[key][matrix_type]
            print(f"{key} {matrix_type}: {type(matrix)} shape {matrix.shape}")

print("\n=== Success! All neighbor methods work ===")

# print(adata.X)

fig, ax = plt.subplots(1, 5, figsize=(20, 4))

sc.pl.umap(
    adata,
    color='louvain',
    ax=ax[0],
    show=False
    )

sc.tl.umap(
    adata, neighbors_key='treeclust_neighbors'
    )  # Just to visualize the data

sc.pl.umap(
    adata,
    color='louvain',
    ax=ax[1],
    show=False
    )

sc.tl.umap(
    adata, neighbors_key='consensus_neighbors'
    )  # Just to visualize the data

sc.pl.umap(
    adata,
    color='louvain',
    ax=ax[2],
    show=False
    )

sc.tl.umap(
    adata, neighbors_key='mutual_neighbors'
    )  # Just to visualize the data

sc.pl.umap(
    adata,
    color='louvain',
    ax=ax[3],
    show=False
    )

sc.tl.umap(
    adata, neighbors_key='consensus_mutual'
    )  # Just to visualize the data

sc.pl.umap(
    adata,
    color='louvain',
    ax=ax[4],
    show=False
    )

plt.show()