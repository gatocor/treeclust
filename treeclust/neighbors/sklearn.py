"""
Sklearn-compatible KNN functions with CUDA dispatch.

This module provides sklearn-compatible functions that automatically dispatch
to either sklearn or cuML implementations based on CUDA availability and user preferences.
All functions maintain sklearn's API for seamless integration.
"""

import numpy as np
import warnings
from typing import Optional, Union, Literal
from scipy import sparse
import scipy.sparse as sp

# Import sklearn backends (required dependency)
from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors

# Import centralized availability flags
from .. import CUML_AVAILABLE

# Import cuML backends conditionally
if CUML_AVAILABLE:
    try:
        import cupy as cp
        from cuml.neighbors import NearestNeighbors as CumlNearestNeighbors
    except ImportError:
        CUML_AVAILABLE = False

def _has_cuda():
    """Check if CUDA is available."""
    try:
        import cupy as cp
        # Try to access a CUDA device
        cp.cuda.Device(0).use()
        return True
    except:
        return False


def _choose_backend(prefer_gpu: bool = True, force_backend: Optional[str] = None) -> str:
    """
    Choose the appropriate backend based on CUDA availability and preferences.
    
    Parameters
    ----------
    prefer_gpu : bool, default=True
        Whether to prefer GPU implementations when available
    force_backend : str, optional
        Force a specific backend ('sklearn', 'cuml')
        
    Returns
    -------
    backend : str
        The chosen backend ('sklearn' or 'cuml')
    """
    if force_backend is not None:
        if force_backend == 'sklearn':
            return 'sklearn'
        elif force_backend == 'cuml' and not CUML_AVAILABLE:
            raise ImportError("cuML backend requested but not available")
        return force_backend
    
    # Auto-select based on preferences and availability
    if prefer_gpu and CUML_AVAILABLE and _has_cuda():
        return 'cuml'
    else:
        return 'sklearn'


def NearestNeighbors(
    n_neighbors: int = 5,
    *,
    radius: float = 1.0,
    algorithm: str = 'auto',
    leaf_size: int = 30,
    metric: str = 'minkowski',
    p: int = 2,
    metric_params: Optional[dict] = None,
    n_jobs: Optional[int] = None,
    prefer_gpu: bool = True,
    backend: Optional[str] = None
):
    """
    Nearest Neighbors with automatic backend selection.
    
    Automatically dispatches to sklearn or cuML NearestNeighbors based on CUDA
    availability and user preferences while maintaining sklearn's API.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use
    radius : float, default=1.0
        Range of parameter space to use by default for radius_neighbors queries
    algorithm : str, default='auto'
        Algorithm used to compute the nearest neighbors (sklearn only)
    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree (sklearn only)
    metric : str, default='minkowski'
        Distance metric to use. Note: cuML supports limited metrics
    p : int, default=2
        Parameter for the Minkowski metric
    metric_params : dict, optional
        Additional keyword arguments for the metric function
    n_jobs : int, optional
        Number of parallel jobs (sklearn only)
    prefer_gpu : bool, default=True
        Whether to prefer GPU implementation when available
    backend : str, optional
        Force specific backend ('sklearn', 'cuml')
        
    Returns
    -------
    nn : NearestNeighbors
        NearestNeighbors estimator with sklearn-compatible API
        
    Examples
    --------
    >>> nn = NearestNeighbors(n_neighbors=5)
    >>> nn.fit(X)
    >>> distances, indices = nn.kneighbors(X)
    """
    chosen_backend = _choose_backend(prefer_gpu=prefer_gpu, force_backend=backend)
    
    if chosen_backend == 'sklearn':
        sklearn_params = {
            'n_neighbors': n_neighbors,
            'radius': radius,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'metric': metric,
            'p': p
        }
        if metric_params is not None:
            sklearn_params['metric_params'] = metric_params
        if n_jobs is not None:
            sklearn_params['n_jobs'] = n_jobs
            
        return SklearnNearestNeighbors(**sklearn_params)
        
    elif chosen_backend == 'cuml':
        # cuML has different parameters, map compatible ones
        cuml_params = {
            'n_neighbors': n_neighbors,
            'metric': metric if metric in ['euclidean', 'l2', 'manhattan', 'l1', 'cosine'] else 'euclidean'
        }
        
        if metric not in ['euclidean', 'l2', 'manhattan', 'l1', 'cosine']:
            warnings.warn(f"cuML doesn't support metric '{metric}', using 'euclidean' instead")
            
        return CumlNearestNeighbors(**cuml_params)
    else:
        raise ValueError(f"Unknown backend: {chosen_backend}")

def get_backend_info() -> dict:
    """
    Get information about available backends.
    
    Returns
    -------
    info : dict
        Dictionary containing backend availability information
    """
    return {
        'sklearn_available': True,
        'cuml_available': CUML_AVAILABLE,
        'cuda_available': _has_cuda(),
        'recommended_backend': 'cuml' if (CUML_AVAILABLE and _has_cuda()) else 'sklearn'
    }

def list_available_algorithms() -> dict:
    """
    List available algorithms for each backend.
    
    Returns
    -------
    algorithms : dict
        Dictionary showing available algorithms for each backend
    """
    return {
        'sklearn': {
            'NearestNeighbors': True,
        },
        'cuml': {
            'NearestNeighbors': CUML_AVAILABLE,
        }
    }