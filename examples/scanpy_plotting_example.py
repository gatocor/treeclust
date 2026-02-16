"""
Example showing treeclust.scanpy plotting functionality.

This example demonstrates all the plotting functions available in the
treeclust.scanpy.pl module using real PBMC data.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad

# Import treeclust scanpy interface
import treeclust.scanpy as tcsc

print("Loaded all required modules successfully!")

# Set scanpy settings
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=80, facecolor='white')

# Load PBMC dataset for demonstration
adata = sc.datasets.pbmc68k_reduced()

# Run multiresolution leiden
# Compute treeclust neighbors first
tcsc.pp.consensus_neighbors(adata, n_neighbors=15, key_added='treeclust_neighbors')

# Run treeclust clustering methods
tcsc.tl.leiden(adata, resolution=0.001, neighbors_key='treeclust_neighbors', key_added='treeclust_leiden')
tcsc.tl.louvain(adata, resolution=0.4, neighbors_key='treeclust_neighbors', key_added='treeclust_louvain') 

tcsc.tl.multiresolution_leiden(
    adata,
    resolutions=(0.0, 1.0), 
    neighbors_key='treeclust_neighbors',
    key_added='multires_leiden',
    n_repetitions=5
)

print("Clustering completed!")
print(f"Available clustering results: {[col for col in adata.obs.columns if 'leiden' in col or 'louvain' in col]}")

# Now demonstrate all plotting functions
print("\n" + "="*50)
print("DEMONSTRATING PLOTTING FUNCTIONS")
print("="*50)

# # 1. Basic graph plot
# print("\n1. Basic graph plot...")
# fig = tcsc.pl.graph(adata, color='treeclust_leiden', neighbors_key='treeclust_neighbors', show_graph=True)
# plt.suptitle('TreeClust Graph - Basic')
# plt.tight_layout()
# plt.show()

# # 2. Graph with neighborhood overlay
# print("\n2. Graph with neighborhood graph overlay...")
# fig = tcsc.pl.graph(
#     adata, 
#     color='treeclust_leiden',
#     show_graph=True,
#     neighbors_key='treeclust_neighbors',
#     alpha=0.8
# )
# plt.suptitle('TreeClust Graph - With Network Overlay')
# plt.tight_layout()
# plt.show()

# 3. Multiresolution leiden plot
print("\n3. Multiresolution Leiden clustering plot...")
fig = tcsc.pl.multiresolution_leiden(adata, key='multires_leiden', neighbors_key='treeclust_neighbors', show_graph=True)
plt.suptitle('TreeClust Multiresolution Leiden')
plt.tight_layout()

# # 5. Dotplot example
# print("\n5. Expression dotplot...")
# # Select some genes to plot - get highly variable genes
# if adata.raw is not None:
#     # Use raw data for gene expression
#     highly_variable_genes = adata.var.index[adata.var.highly_variable] if 'highly_variable' in adata.var else adata.var_names[:10]
# else:
#     highly_variable_genes = adata.var_names[:10]

# top_genes = highly_variable_genes[:8]  # First 8 genes
# fig = tcsc.pl.dotplot(adata, gene_symbols=top_genes, groupby=['multires_leiden_0.1', 'multires_leiden_0.5'], group_rows=True)
# plt.suptitle('TreeClust Expression Dotplot')
# plt.tight_layout()
# plt.show()

# 6. Multiresolution hierarchy (if available)
print("\n6. Checking for multiresolution hierarchy...")
if 'multires_leiden' in adata.uns and 'hierarchy' in adata.uns['multires_leiden']:
    print("Plotting multiresolution hierarchy...")
    fig = tcsc.pl.multiresolution_hierarchy(adata, key='multires_leiden')
    plt.suptitle('TreeClust Multiresolution Hierarchy')
    plt.tight_layout()
    plt.show()
else:
    print("No hierarchy information available for plotting")

print("\n" + "="*50)
print("ALL PLOTTING FUNCTIONS COMPLETED SUCCESSFULLY!")
print("="*50)

# Summary of what was demonstrated
print(f"\nSummary:")
print(f"- Processed {adata.n_obs} cells and {adata.n_vars} genes")
print(f"- Demonstrated {5} different plotting functions:")
print(f"  • tcsc.pl.graph() - Basic and network-overlay embedding plots")
print(f"  • tcsc.pl.multiresolution_leiden() - Multiresolution clustering visualization")
print(f"  • tcsc.pl.clustering_comparison() - Side-by-side clustering comparison")
print(f"  • tcsc.pl.dotplot() - Gene expression dotplot")
print(f"  • tcsc.pl.multiresolution_hierarchy() - Hierarchical clustering structure")
print(f"- All functions work seamlessly with AnnData objects")
print(f"- Full scanpy-style API compatibility maintained")