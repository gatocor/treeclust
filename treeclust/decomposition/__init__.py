"""
Decomposition module for treeclust.

This module provides dimensionality reduction algorithms with automatic
backend selection for optimal performance.
"""

# Import available classes
try:
    from .vae import VAE
    _HAS_VAE = True
except ImportError:
    _HAS_VAE = False
    VAE = None

try:
    from .tsne import TSNE
    _HAS_TSNE = True
except ImportError:
    _HAS_TSNE = False
    TSNE = None

try:
    from .umap import UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False
    UMAP = None

try:
    from .scvi import ScviWrapper
    _HAS_SCVI = True
except ImportError:
    _HAS_SCVI = False
    ScviWrapper = None

# Create __all__ dynamically based on what's available
__all__ = []
if _HAS_VAE:
    __all__.append('VAE')
if _HAS_TSNE:
    __all__.append('TSNE')
if _HAS_UMAP:
    __all__.append('UMAP')
if _HAS_SCVI:
    __all__.append('ScviWrapper')

# Import UMAP and TSNE classes if available
try:
    from .umap import UMAP as UMAPClass
except ImportError:
    UMAPClass = None

try:
    from .tsne import TSNE as TSNEClass
except ImportError:
    TSNEClass = None

# Import sklearn-compatible dispatch functions
try:
    from .sklearn import (
        PCA, TruncatedSVD, FastICA, TSNE, UMAP,
        get_backend_info, list_available_algorithms
    )
except ImportError:
    # If sklearn or other dependencies aren't available, create placeholder functions
    def PCA(*args, **kwargs):
        raise ImportError("sklearn or other dependencies not available for PCA")
    
    def TruncatedSVD(*args, **kwargs):
        raise ImportError("sklearn or other dependencies not available for TruncatedSVD")
    
    def FastICA(*args, **kwargs):
        raise ImportError("sklearn or other dependencies not available for FastICA")
    
    def TSNE(*args, **kwargs):
        raise ImportError("sklearn or other dependencies not available for TSNE")
    
    def UMAP(*args, **kwargs):
        raise ImportError("sklearn or other dependencies not available for UMAP")
    
    def get_backend_info():
        return {}
    
    def list_available_algorithms():
        return {}

# Expose both class and function versions for TSNE/UMAP
if TSNEClass is not None:
    globals()['TSNEClass'] = TSNEClass
if UMAPClass is not None:
    globals()['UMAPClass'] = UMAPClass

__all__ = [
    'VAE',
    'PCA', 'TruncatedSVD', 'FastICA', 'TSNE', 'UMAP',
    'get_backend_info', 'list_available_algorithms'
]

# Add class versions to __all__ if available
if TSNEClass is not None:
    __all__.append('TSNEClass')
if UMAPClass is not None:
    __all__.append('UMAPClass')