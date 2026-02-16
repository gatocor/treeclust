import scanpy as sc
import treeclust.scanpy as tcsc
import numpy as np

# Load example data
adata = sc.datasets.pbmc3k_processed()
print(f"Data shape: {adata.shape}")
print(f"Original obs columns: {list(adata.obs.columns)}")

# First compute neighbors (required for clustering)
print("\n=== Computing neighbors ===")
tcsc.pp.mutual_neighbors(adata, n_neighbors=15)
print(f"Neighbors computed: {list(adata.uns.keys())}")

# Check that adjacency information is available
print(f"Available in adata.obsp: {list(adata.obsp.keys())}")
print(f"Available in adata.uns: {list(adata.uns.keys())}")

# Test different clustering methods
print("\n=== Testing treeclust scanpy-style clustering ===")

# 1. Standard Leiden clustering (like scanpy.tl.leiden)
print("\n1. Leiden clustering...")
tcsc.tl.leiden(adata, resolution=0.1, key_added='treeclust_leiden')
print(f"Leiden clusters: {adata.obs['treeclust_leiden'].nunique()} clusters")
print(f"Cluster distribution: {adata.obs['treeclust_leiden'].value_counts().sort_index()}")

# 2. Louvain clustering
print("\n2. Louvain clustering...")
tcsc.tl.louvain(adata, resolution=0.5, key_added='treeclust_louvain')
print(f"Louvain clusters: {adata.obs['treeclust_louvain'].nunique()} clusters")
print(f"Cluster distribution: {adata.obs['treeclust_louvain'].value_counts().sort_index()}")

# 3. Multiresolution Leiden clustering
print("\n3. Multiresolution Leiden clustering...")
tcsc.tl.multiresolution_leiden(
    adata, 
    resolutions=(0.1, 1.0), 
    key_added='multires_leiden'
)
print(f"Multiresolution results: {[col for col in adata.obs.columns if 'multires_leiden' in col]}")
print(f"Hierarchy info: {list(adata.uns['multires_leiden'].keys())}")

for res in [0.2, 0.5, 0.8]:
    col = f'multires_leiden_{res}'
    if col in adata.obs.columns:
        print(f"  Resolution {res}: {adata.obs[col].nunique()} clusters")

# 4. Multiresolution Louvain clustering
print("\n4. Multiresolution Louvain clustering...")
tcsc.tl.multiresolution_louvain(
    adata, 
    resolutions=[0.2, 0.5, 0.8], 
    key_added='multires_louvain'
)
print(f"Multiresolution Louvain results: {[col for col in adata.obs.columns if 'multires_louvain' in col]}")

for res in [0.2, 0.5, 0.8]:
    col = f'multires_louvain_{res}'
    if col in adata.obs.columns:
        print(f"  Resolution {res}: {adata.obs[col].nunique()} clusters")

# Show final clustering results
print("\n=== Final clustering comparison ===")
clustering_cols = [col for col in adata.obs.columns if any(x in col for x in ['leiden', 'louvain'])]
for col in clustering_cols:
    n_clusters = adata.obs[col].nunique()
    print(f"{col}: {n_clusters} clusters")

print("\n=== Success! All clustering methods work ===")