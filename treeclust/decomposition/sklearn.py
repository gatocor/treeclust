"""
Sklearn-compatible decomposition functions with CUDA dispatch.

This module provides sklearn-compatible functions that automatically dispatch
to either sklearn or cuML implementations based on CUDA availability and user preferences.
All functions maintain sklearn's API for seamless integration.
"""

import numpy as np
import warnings
from typing import Optional, Union, Dict, Any

# Import sklearn backends (required dependency)
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.decomposition import TruncatedSVD as SklearnTruncatedSVD
from sklearn.decomposition import FastICA as SklearnFastICA
from sklearn.manifold import TSNE as SklearnTSNE
from sklearn.base import BaseEstimator, TransformerMixin

# Import centralized availability flags
from .. import CUML_AVAILABLE, UMAP_AVAILABLE, CUML_UMAP_AVAILABLE

# Import cuML backends conditionally
if CUML_AVAILABLE:
    try:
        import cupy as cp
        from cuml.decomposition import PCA as CumlPCA
        from cuml.decomposition import TruncatedSVD as CumlTruncatedSVD
        from cuml.manifold import TSNE as CumlTSNE
    except ImportError:
        CUML_AVAILABLE = False

# Import UMAP backends conditionally
if UMAP_AVAILABLE:
    try:
        import umap
    except ImportError:
        UMAP_AVAILABLE = False

if CUML_UMAP_AVAILABLE:
    try:
        from cuml.manifold import UMAP as CumlUMAP
    except ImportError:
        CUML_UMAP_AVAILABLE = False


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


def PCA(
    n_components: Optional[int] = None,
    *,
    copy: bool = True,
    whiten: bool = False,
    svd_solver: str = 'auto',
    tol: float = 0.0,
    iterated_power: str = 'auto',
    random_state: Optional[int] = None,
    prefer_gpu: bool = True,
    backend: Optional[str] = None
) -> BaseEstimator:
    """
    Principal Component Analysis (PCA) with automatic backend selection.
    
    Automatically dispatches to sklearn or cuML PCA based on CUDA availability
    and user preferences while maintaining sklearn's API.
    
    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If n_components is not set, all components are kept
    copy : bool, default=True
        If False, data passed to fit are overwritten and running fit(X).transform(X)
        will not yield the expected results
    whiten : bool, default=False
        When True, the components_ vectors are multiplied by the square root of
        n_samples and then divided by the singular values to ensure uncorrelated outputs
    svd_solver : str, default='auto'
        SVD solver to use. sklearn only parameter
    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver
    iterated_power : str or int, default='auto'
        Number of iterations for the power method computed by svd_solver
    random_state : int, optional
        Random seed for reproducible results
    prefer_gpu : bool, default=True
        Whether to prefer GPU implementation when available
    backend : str, optional
        Force specific backend ('sklearn', 'cuml')
        
    Returns
    -------
    pca : BaseEstimator
        PCA transformer with sklearn-compatible API
        
    Examples
    --------
    >>> pca = PCA(n_components=2)
    >>> X_reduced = pca.fit_transform(X)
    """
    chosen_backend = _choose_backend(prefer_gpu=prefer_gpu, force_backend=backend)
    
    if chosen_backend == 'sklearn':
        return SklearnPCA(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state
        )
    elif chosen_backend == 'cuml':
        # cuML PCA has different parameters, map compatible ones
        cuml_params = {
            'n_components': n_components,
            'copy': copy,
            'whiten': whiten,
            'random_state': random_state
        }
        return CumlPCA(**cuml_params)
    else:
        raise ValueError(f"Unknown backend: {chosen_backend}")


def TruncatedSVD(
    n_components: int = 2,
    *,
    algorithm: str = 'randomized',
    n_iter: int = 5,
    random_state: Optional[int] = None,
    tol: float = 0.0,
    prefer_gpu: bool = True,
    backend: Optional[str] = None
) -> BaseEstimator:
    """
    Truncated Singular Value Decomposition with automatic backend selection.
    
    Parameters
    ----------
    n_components : int, default=2
        Desired dimensionality of output data
    algorithm : str, default='randomized'
        SVD solver to use. sklearn only parameter
    n_iter : int, default=5
        Number of iterations for randomized SVD solver
    random_state : int, optional
        Random seed for reproducible results
    tol : float, default=0.0
        Tolerance for ARPACK
    prefer_gpu : bool, default=True
        Whether to prefer GPU implementation when available
    backend : str, optional
        Force specific backend ('sklearn', 'cuml')
        
    Returns
    -------
    svd : BaseEstimator
        TruncatedSVD transformer with sklearn-compatible API
    """
    chosen_backend = _choose_backend(prefer_gpu=prefer_gpu, force_backend=backend)
    
    if chosen_backend == 'sklearn':
        return SklearnTruncatedSVD(
            n_components=n_components,
            algorithm=algorithm,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol
        )
    elif chosen_backend == 'cuml':
        # cuML TruncatedSVD has different parameters
        cuml_params = {
            'n_components': n_components,
            'n_iter': n_iter,
            'random_state': random_state,
            'tol': tol
        }
        return CumlTruncatedSVD(**cuml_params)
    else:
        raise ValueError(f"Unknown backend: {chosen_backend}")


def FastICA(
    n_components: Optional[int] = None,
    *,
    algorithm: str = 'parallel',
    whiten: bool = True,
    fun: str = 'logcosh',
    fun_args: Optional[Dict] = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    w_init: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    prefer_gpu: bool = True,
    backend: Optional[str] = None
) -> BaseEstimator:
    """
    Independent Component Analysis with automatic backend selection.
    
    Note: cuML may not have FastICA, will default to sklearn.
    
    Parameters
    ----------
    n_components : int, optional
        Number of components to use
    algorithm : str, default='parallel'
        Algorithm to use ('parallel', 'deflation')
    whiten : bool, default=True
        Whether to perform whitening
    fun : str, default='logcosh'
        The functional form of the G function used in FastICA
    fun_args : dict, optional
        Arguments to send to the functional form
    max_iter : int, default=200
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance on update at each iteration
    w_init : array-like, optional
        Initial un-mixing array
    random_state : int, optional
        Random seed for reproducible results
    prefer_gpu : bool, default=True
        Whether to prefer GPU implementation when available
    backend : str, optional
        Force specific backend ('sklearn')
        
    Returns
    -------
    ica : BaseEstimator
        FastICA transformer with sklearn-compatible API
    """
    # cuML doesn't have FastICA, force sklearn
    if backend == 'cuml':
        warnings.warn("cuML doesn't have FastICA, using sklearn instead")
        backend = 'sklearn'
    
    chosen_backend = _choose_backend(prefer_gpu=False, force_backend=backend or 'sklearn')
    
    if chosen_backend == 'sklearn':
        return SklearnFastICA(
            n_components=n_components,
            algorithm=algorithm,
            whiten=whiten,
            fun=fun,
            fun_args=fun_args,
            max_iter=max_iter,
            tol=tol,
            w_init=w_init,
            random_state=random_state
        )
    else:
        raise ValueError("FastICA only available with sklearn backend")


def TSNE(
    n_components: int = 2,
    *,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: Union[str, float] = 'warn',
    n_iter: int = 1000,
    n_iter_without_progress: int = 300,
    min_grad_norm: float = 1e-7,
    metric: str = 'euclidean',
    init: str = 'random',
    verbose: int = 0,
    random_state: Optional[int] = None,
    method: str = 'barnes_hut',
    angle: float = 0.5,
    n_jobs: Optional[int] = None,
    prefer_gpu: bool = True,
    backend: Optional[str] = None
) -> BaseEstimator:
    """
    t-SNE with automatic backend selection.
    
    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space
    perplexity : float, default=30.0
        The perplexity parameter
    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are
    learning_rate : float or 'warn', default='warn'
        The learning rate for t-SNE
    n_iter : int, default=1000
        Maximum number of iterations
    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress
    min_grad_norm : float, default=1e-7
        Minimum gradient norm
    metric : str, default='euclidean'
        Distance metric to use
    init : str, default='random'
        Initialization method
    verbose : int, default=0
        Verbosity level
    random_state : int, optional
        Random seed for reproducible results
    method : str, default='barnes_hut'
        Algorithm to use
    angle : float, default=0.5
        Trade-off between speed and accuracy for Barnes-Hut
    n_jobs : int, optional
        Number of parallel jobs (sklearn only)
    prefer_gpu : bool, default=True
        Whether to prefer GPU implementation when available
    backend : str, optional
        Force specific backend ('sklearn', 'cuml')
        
    Returns
    -------
    tsne : BaseEstimator
        t-SNE transformer with sklearn-compatible API
    """
    chosen_backend = _choose_backend(prefer_gpu=prefer_gpu, force_backend=backend)
    
    if chosen_backend == 'sklearn':
        sklearn_params = {
            'n_components': n_components,
            'perplexity': perplexity,
            'early_exaggeration': early_exaggeration,
            'learning_rate': learning_rate,
            'n_iter': n_iter,
            'n_iter_without_progress': n_iter_without_progress,
            'min_grad_norm': min_grad_norm,
            'metric': metric,
            'init': init,
            'verbose': verbose,
            'random_state': random_state,
            'method': method,
            'angle': angle
        }
        if n_jobs is not None:
            sklearn_params['n_jobs'] = n_jobs
        
        return SklearnTSNE(**sklearn_params)
    elif chosen_backend == 'cuml':
        # cuML t-SNE has different parameters, map compatible ones
        cuml_params = {
            'n_components': n_components,
            'perplexity': perplexity,
            'early_exaggeration': early_exaggeration,
            'learning_rate': learning_rate,
            'n_iter': n_iter,
            'min_grad_norm': min_grad_norm,
            'metric': metric,
            'init': init,
            'verbose': verbose,
            'random_state': random_state,
            'method': method,
            'angle': angle
        }
        # Filter parameters that cuML supports
        return CumlTSNE(**{k: v for k, v in cuml_params.items() 
                          if k in CumlTSNE.__init__.__code__.co_varnames})
    else:
        raise ValueError(f"Unknown backend: {chosen_backend}")


def UMAP(
    n_components: int = 2,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    metric_kwds: Optional[Dict] = None,
    output_metric: str = 'euclidean',
    output_metric_kwds: Optional[Dict] = None,
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    init: str = 'spectral',
    spread: float = 1.0,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: int = 1,
    repulsion_strength: float = 1.0,
    negative_sample_rate: int = 5,
    transform_queue_size: float = 4.0,
    a: Optional[float] = None,
    b: Optional[float] = None,
    random_state: Optional[int] = None,
    angular_rp_forest: bool = False,
    target_n_neighbors: int = -1,
    target_metric: str = 'categorical',
    target_metric_kwds: Optional[Dict] = None,
    target_weight: float = 0.5,
    transform_seed: int = 42,
    verbose: bool = False,
    prefer_gpu: bool = True,
    backend: Optional[str] = None
) -> BaseEstimator:
    """
    UMAP with automatic backend selection.
    
    Parameters
    ----------
    n_components : int, default=2
        Dimension of the space to embed into
    n_neighbors : int, default=15
        Number of neighboring sample points used for manifold approximation
    min_dist : float, default=0.1
        Effective minimum distance between embedded points
    metric : str, default='euclidean'
        Distance metric to use
    metric_kwds : dict, optional
        Arguments to pass to the metric
    output_metric : str, default='euclidean'
        Metric for output space
    output_metric_kwds : dict, optional
        Arguments for output metric
    n_epochs : int, optional
        Number of training epochs
    learning_rate : float, default=1.0
        Initial learning rate
    init : str, default='spectral'
        Initialization method
    spread : float, default=1.0
        Effective scale of embedded points
    set_op_mix_ratio : float, default=1.0
        Set operation mix ratio
    local_connectivity : int, default=1
        Local connectivity parameter
    repulsion_strength : float, default=1.0
        Weighting for negative samples
    negative_sample_rate : int, default=5
        Number of negative samples per positive sample
    transform_queue_size : float, default=4.0
        Transform queue size
    a : float, optional
        More specific parameters controlling the embedding
    b : float, optional
        More specific parameters controlling the embedding
    random_state : int, optional
        Random seed for reproducible results
    angular_rp_forest : bool, default=False
        Whether to use angular random projection forest
    target_n_neighbors : int, default=-1
        Number of nearest neighbors for target
    target_metric : str, default='categorical'
        Target metric for supervised dimension reduction
    target_metric_kwds : dict, optional
        Arguments for target metric
    target_weight : float, default=0.5
        Weight between data and target topology
    transform_seed : int, default=42
        Random seed for transform operations
    verbose : bool, default=False
        Verbosity flag
    prefer_gpu : bool, default=True
        Whether to prefer GPU implementation when available
    backend : str, optional
        Force specific backend ('umap', 'cuml')
        
    Returns
    -------
    umap : BaseEstimator
        UMAP transformer with sklearn-compatible API
    """
    # Handle backend selection for UMAP
    if backend is None:
        if prefer_gpu and CUML_UMAP_AVAILABLE and _has_cuda():
            chosen_backend = 'cuml'
        elif UMAP_AVAILABLE:
            chosen_backend = 'umap'
        else:
            raise ImportError("No UMAP implementation available")
    else:
        if backend == 'umap' and not UMAP_AVAILABLE:
            raise ImportError("umap-learn not available")
        elif backend == 'cuml' and not CUML_UMAP_AVAILABLE:
            raise ImportError("cuML UMAP not available")
        chosen_backend = backend
    
    if chosen_backend == 'umap':
        umap_params = {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
            'learning_rate': learning_rate,
            'init': init,
            'spread': spread,
            'set_op_mix_ratio': set_op_mix_ratio,
            'local_connectivity': local_connectivity,
            'repulsion_strength': repulsion_strength,
            'negative_sample_rate': negative_sample_rate,
            'transform_queue_size': transform_queue_size,
            'random_state': random_state,
            'angular_rp_forest': angular_rp_forest,
            'target_n_neighbors': target_n_neighbors,
            'target_metric': target_metric,
            'target_weight': target_weight,
            'transform_seed': transform_seed,
            'verbose': verbose
        }
        
        # Add optional parameters
        if metric_kwds is not None:
            umap_params['metric_kwds'] = metric_kwds
        if output_metric_kwds is not None:
            umap_params['output_metric_kwds'] = output_metric_kwds
        if n_epochs is not None:
            umap_params['n_epochs'] = n_epochs
        if a is not None:
            umap_params['a'] = a
        if b is not None:
            umap_params['b'] = b
        if target_metric_kwds is not None:
            umap_params['target_metric_kwds'] = target_metric_kwds
        
        return umap.UMAP(**umap_params)
        
    elif chosen_backend == 'cuml':
        # cuML UMAP parameters (subset of umap-learn)
        cuml_params = {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
            'learning_rate': learning_rate,
            'init': init,
            'spread': spread,
            'set_op_mix_ratio': set_op_mix_ratio,
            'local_connectivity': local_connectivity,
            'repulsion_strength': repulsion_strength,
            'negative_sample_rate': negative_sample_rate,
            'transform_queue_size': transform_queue_size,
            'random_state': random_state,
            'angular_rp_forest': angular_rp_forest,
            'target_n_neighbors': target_n_neighbors,
            'target_metric': target_metric,
            'target_weight': target_weight,
            'verbose': verbose
        }
        
        # Filter parameters that cuML supports
        return CumlUMAP(**{k: v for k, v in cuml_params.items() 
                          if k in CumlUMAP.__init__.__code__.co_varnames})
    else:
        raise ValueError(f"Unknown backend: {chosen_backend}")


def get_backend_info() -> Dict[str, Any]:
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
        'umap_available': UMAP_AVAILABLE,
        'cuml_umap_available': CUML_UMAP_AVAILABLE,
        'cuda_available': _has_cuda(),
        'recommended_backend': 'cuml' if (CUML_AVAILABLE and _has_cuda()) else 'sklearn'
    }


def list_available_algorithms() -> Dict[str, Dict[str, bool]]:
    """
    List available algorithms for each backend.
    
    Returns
    -------
    algorithms : dict
        Dictionary showing available algorithms for each backend
    """
    return {
        'sklearn': {
            'PCA': True,
            'TruncatedSVD': True,
            'FastICA': True,
            'TSNE': True,
            'UMAP': UMAP_AVAILABLE
        },
        'cuml': {
            'PCA': CUML_AVAILABLE,
            'TruncatedSVD': CUML_AVAILABLE,
            'FastICA': False,  # cuML doesn't have FastICA
            'TSNE': CUML_AVAILABLE,
            'UMAP': CUML_UMAP_AVAILABLE
        }
    }