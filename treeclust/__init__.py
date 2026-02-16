"""
TreeClust: Package for robust clustering in a hierarchical manner.

TreeClust provides various clustering algorithms with sklearn-style interfaces,
consistency checking for stochastic methods, and specialized parameter tuning
capabilities for robust clustering analysis.

Individual modules can be imported directly:
    from treeclust.clustering import Leiden, Louvain
    from treeclust.neighbors import ConsensusKNeighbors
    from treeclust.plotting import plot_graph
    from treeclust.metrics import connectivity_probability
    from treeclust.utils import propagate_labels, propagate_labels_iterative
"""

__version__ = "0.0.1"

# Centralized availability checking for all optional dependencies
# These flags are imported throughout the treeclust package

# sklearn is a required dependency (no check needed)
SKLEARN_AVAILABLE = True

# GPU acceleration libraries
try:
    import cuml
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

try:
    import cugraph
    import cudf
    CURAPIDS_AVAILABLE = True
except ImportError:
    CURAPIDS_AVAILABLE = False

# Deep learning libraries
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Graph libraries
try:
    import igraph
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

# UMAP library
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Check for cuML UMAP specifically
try:
    from cuml.manifold import UMAP as CumlUMAP
    CUML_UMAP_AVAILABLE = True
except ImportError:
    CUML_UMAP_AVAILABLE = False

# Dimensionality reduction availability (depends on optional packages)
DIM_REDUCTION_AVAILABLE = any([
    TORCH_AVAILABLE,  # For VAE
    True,  # sklearn always available for PCA, t-SNE
    UMAP_AVAILABLE,   # For UMAP
])