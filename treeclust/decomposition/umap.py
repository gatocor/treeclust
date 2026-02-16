"""
Sklearn-style UMAP implementation with backend dispatch.

This module provides a sklearn-compatible UMAP class that can dispatch to
different backends while maintaining a consistent API.
"""

import numpy as np
import warnings
from typing import Optional, Union, Literal, Any
from sklearn.base import BaseEstimator, TransformerMixin

# Import centralized availability flags
from .. import UMAP_AVAILABLE, CUML_AVAILABLE

# Import UMAP backend conditionally
if UMAP_AVAILABLE:
    try:
        import umap
    except ImportError:
        UMAP_AVAILABLE = False

# Import cuML UMAP backend conditionally
if CUML_AVAILABLE:
    try:
        from cuml.manifold import UMAP as CumlUMAP
    except ImportError:
        CUML_AVAILABLE = False

class UMAP(BaseEstimator, TransformerMixin):
    """
    UMAP (Uniform Manifold Approximation and Projection) with backend dispatch.
    
    This class provides a sklearn-compatible interface for UMAP that can
    automatically choose between different backends (umap-learn, cuML) based
    on availability and user preferences.
    
    Parameters
    ----------
    n_components : int, default=2
        The dimension of the space to embed into
    n_neighbors : int, default=15
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation
    min_dist : float, default=0.1
        The effective minimum distance between embedded points
    metric : str or callable, default='euclidean'
        The metric to use to compute distances in high dimensional space
    metric_kwds : dict, optional
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance
    output_metric : str or callable, default='euclidean'
        The metric used to measure distance for a nearest neighbors search
    output_metric_kwds : dict, optional
        Arguments to pass on to the output metric
    n_epochs : int, optional
        The number of training epochs to be used in optimizing the low dimensional
        embedding
    learning_rate : float, default=1.0
        The initial learning rate for the embedding optimization
    init : str or np.array, default='spectral'
        How to initialize the low dimensional embedding
    spread : float, default=1.0
        The effective scale of embedded points
    set_op_mix_ratio : float, default=1.0
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial set
    local_connectivity : int, default=1
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level
    repulsion_strength : float, default=1.0
        Weighting applied to negative samples in low dimensional embedding
        optimization
    negative_sample_rate : int, default=5
        The number of negative edge/2-cell samples to use per positive edge/2-cell
        sample in optimizing the low dimensional embedding
    transform_queue_size : float, default=4.0
        For transform operations (embedding new points using a trained model),
        this will control how aggressively to search for nearest neighbors
    a : float, optional
        More specific parameters controlling the embedding
    b : float, optional
        More specific parameters controlling the embedding
    random_state : int, optional
        Random seed used for the stochastic aspects of the transform operation
    angular_rp_forest : bool, default=False
        Whether to use an angular random projection forest to initialise
        the approximate nearest neighbor search
    target_n_neighbors : int, default=-1
        The number of nearest neighbors to use to construct the target simplcial
        set when using supervised dimension reduction
    target_metric : str or callable, default='categorical'
        The metric used to measure distance for a target array is using supervised
        dimension reduction
    target_metric_kwds : dict, optional
        Dictionary of arguments to pass on to the target metric when performing
        supervised dimension reduction
    target_weight : float, default=0.5
        weighting factor between data topology and target topology
    transform_seed : int, default=42
        Random seed used for the stochastic aspects of the transform operation
    verbose : bool, default=False
        Controls verbosity of logging
    unique : bool, default=False
        Controls if the rows of your data should be uniqued before being
        embedded
    densmap : bool, default=False
        Specifies whether the density-augmented objective of densMAP should be used
        for optimization
    dens_lambda : float, default=2.0
        Controls the regularization weight of the density correlation term in densMAP
    dens_frac : float, default=0.3
        Controls the fraction of epochs (between 0 and 1) where the
        density-augmented objective is used in densMAP
    dens_var_shift : float, default=0.1
        A small constant added to the variance of local densities in the
        embedding when operating in densMAP mode
    output_dens : bool, default=False
        Determines whether the local densities are computed and returned in
        addition to the embedding
    disconnection_distance : float, optional
        Disconnect any vertices of distance greater than or equal to
        disconnection_distance when approximating the manifold via local
        linear patches
    precomputed_knn : tuple, optional
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time
    backend : str, optional
        Backend to use ('umap', 'cuml', 'auto')
    prefer_gpu : bool, default=False
        Whether to prefer GPU implementations when available
    
    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the embedding vectors
    n_features_in_ : int
        Number of features seen during fit
    feature_names_in_ : ndarray of shape (n_features_in_,), optional
        Names of features seen during fit
    """
    
    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        metric_kwds: Optional[dict] = None,
        output_metric: str = 'euclidean',
        output_metric_kwds: Optional[dict] = None,
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
        target_metric_kwds: Optional[dict] = None,
        target_weight: float = 0.5,
        transform_seed: int = 42,
        verbose: bool = False,
        unique: bool = False,
        densmap: bool = False,
        dens_lambda: float = 2.0,
        dens_frac: float = 0.3,
        dens_var_shift: float = 0.1,
        output_dens: bool = False,
        disconnection_distance: Optional[float] = None,
        precomputed_knn: Optional[tuple] = None,
        backend: Optional[str] = None,
        prefer_gpu: bool = False
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.output_metric = output_metric
        self.output_metric_kwds = output_metric_kwds
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.spread = spread
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.transform_queue_size = transform_queue_size
        self.a = a
        self.b = b
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.verbose = verbose
        self.unique = unique
        self.densmap = densmap
        self.dens_lambda = dens_lambda
        self.dens_frac = dens_frac
        self.dens_var_shift = dens_var_shift
        self.output_dens = output_dens
        self.disconnection_distance = disconnection_distance
        self.precomputed_knn = precomputed_knn
        self.backend = backend
        self.prefer_gpu = prefer_gpu
        
        # Initialize the backend estimator
        self._estimator = None
        self._chosen_backend = None
    
    def _choose_backend(self):
        """Choose the appropriate backend based on preferences and availability."""
        if self.backend is not None:
            # Explicit backend requested
            if self.backend == 'umap' and not UMAP_AVAILABLE:
                raise ImportError("umap-learn backend requested but not available")
            elif self.backend == 'cuml' and not CUML_AVAILABLE:
                raise ImportError("cuml backend requested but not available")
            return self.backend
        
        # Auto-select backend
        if self.prefer_gpu and CUML_AVAILABLE:
            return 'cuml'
        elif UMAP_AVAILABLE:
            return 'umap'
        else:
            raise ImportError("No suitable backend available for UMAP")
    
    def _create_estimator(self):
        """Create the appropriate backend estimator."""
        self._chosen_backend = self._choose_backend()
        
        if self._chosen_backend == 'umap':
            # Create umap-learn UMAP
            umap_params = {
                'n_components': self.n_components,
                'n_neighbors': self.n_neighbors,
                'min_dist': self.min_dist,
                'metric': self.metric,
                'learning_rate': self.learning_rate,
                'init': self.init,
                'spread': self.spread,
                'set_op_mix_ratio': self.set_op_mix_ratio,
                'local_connectivity': self.local_connectivity,
                'repulsion_strength': self.repulsion_strength,
                'negative_sample_rate': self.negative_sample_rate,
                'transform_queue_size': self.transform_queue_size,
                'random_state': self.random_state,
                'angular_rp_forest': self.angular_rp_forest,
                'target_n_neighbors': self.target_n_neighbors,
                'target_metric': self.target_metric,
                'target_weight': self.target_weight,
                'transform_seed': self.transform_seed,
                'verbose': self.verbose,
                'unique': self.unique,
                'densmap': self.densmap,
                'dens_lambda': self.dens_lambda,
                'dens_frac': self.dens_frac,
                'dens_var_shift': self.dens_var_shift,
                'output_dens': self.output_dens
            }
            
            # Add optional parameters if they are not None
            if self.metric_kwds is not None:
                umap_params['metric_kwds'] = self.metric_kwds
            if self.output_metric_kwds is not None:
                umap_params['output_metric_kwds'] = self.output_metric_kwds
            if self.n_epochs is not None:
                umap_params['n_epochs'] = self.n_epochs
            if self.a is not None:
                umap_params['a'] = self.a
            if self.b is not None:
                umap_params['b'] = self.b
            if self.target_metric_kwds is not None:
                umap_params['target_metric_kwds'] = self.target_metric_kwds
            if self.disconnection_distance is not None:
                umap_params['disconnection_distance'] = self.disconnection_distance
            if self.precomputed_knn is not None:
                umap_params['precomputed_knn'] = self.precomputed_knn
            
            self._estimator = umap.UMAP(**umap_params)
            
        elif self._chosen_backend == 'cuml':
            # Create cuML UMAP
            cuml_params = {
                'n_components': self.n_components,
                'n_neighbors': self.n_neighbors,
                'min_dist': self.min_dist,
                'metric': self.metric,
                'learning_rate': self.learning_rate,
                'init': self.init,
                'spread': self.spread,
                'set_op_mix_ratio': self.set_op_mix_ratio,
                'local_connectivity': self.local_connectivity,
                'repulsion_strength': self.repulsion_strength,
                'negative_sample_rate': self.negative_sample_rate,
                'transform_queue_size': self.transform_queue_size,
                'random_state': self.random_state,
                'angular_rp_forest': self.angular_rp_forest,
                'target_n_neighbors': self.target_n_neighbors,
                'target_metric': self.target_metric,
                'target_weight': self.target_weight,
                'verbose': self.verbose
            }
            
            # Filter parameters that cuML supports
            self._estimator = CumlUMAP(**{k: v for k, v in cuml_params.items() 
                                        if k in CumlUMAP.__init__.__code__.co_varnames})
    
    def fit(self, X, y=None):
        """
        Fit the model with X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Target values for supervised dimension reduction
            
        Returns
        -------
        self : object
            Returns self
        """
        if self._estimator is None:
            self._create_estimator()
        
        # Fit the estimator
        if y is not None:
            self._estimator.fit(X, y)
        else:
            self._estimator.fit(X)
        
        # Copy attributes from the backend estimator
        if hasattr(self._estimator, 'embedding_'):
            self.embedding_ = self._estimator.embedding_
        if hasattr(self._estimator, 'n_features_in_'):
            self.n_features_in_ = self._estimator.n_features_in_
        if hasattr(self._estimator, 'feature_names_in_'):
            self.feature_names_in_ = self._estimator.feature_names_in_
        
        return self
    
    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Target values for supervised dimension reduction
            
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space
        """
        if self._estimator is None:
            self._create_estimator()
        
        # Use fit_transform from backend
        if y is not None:
            X_transformed = self._estimator.fit_transform(X, y)
        else:
            X_transformed = self._estimator.fit_transform(X)
        
        # Copy attributes from the backend estimator
        if hasattr(self._estimator, 'embedding_'):
            self.embedding_ = self._estimator.embedding_
        if hasattr(self._estimator, 'n_features_in_'):
            self.n_features_in_ = self._estimator.n_features_in_
        if hasattr(self._estimator, 'feature_names_in_'):
            self.feature_names_in_ = self._estimator.feature_names_in_
        
        return X_transformed
    
    def transform(self, X):
        """
        Transform X into the embedded space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform
            
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space
        """
        if self._estimator is None:
            raise ValueError("This UMAP instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        return self._estimator.transform(X)
    
    def inverse_transform(self, X):
        """
        Transform X in the embedded space back to the original space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in embedded space
            
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in original space
        """
        if self._estimator is None:
            raise ValueError("This UMAP instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        if not hasattr(self._estimator, 'inverse_transform'):
            raise AttributeError(f"{self._chosen_backend} backend does not support inverse_transform")
        
        return self._estimator.inverse_transform(X)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values
        """
        params = {
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'metric_kwds': self.metric_kwds,
            'output_metric': self.output_metric,
            'output_metric_kwds': self.output_metric_kwds,
            'n_epochs': self.n_epochs,
            'learning_rate': self.learning_rate,
            'init': self.init,
            'spread': self.spread,
            'set_op_mix_ratio': self.set_op_mix_ratio,
            'local_connectivity': self.local_connectivity,
            'repulsion_strength': self.repulsion_strength,
            'negative_sample_rate': self.negative_sample_rate,
            'transform_queue_size': self.transform_queue_size,
            'a': self.a,
            'b': self.b,
            'random_state': self.random_state,
            'angular_rp_forest': self.angular_rp_forest,
            'target_n_neighbors': self.target_n_neighbors,
            'target_metric': self.target_metric,
            'target_metric_kwds': self.target_metric_kwds,
            'target_weight': self.target_weight,
            'transform_seed': self.transform_seed,
            'verbose': self.verbose,
            'unique': self.unique,
            'densmap': self.densmap,
            'dens_lambda': self.dens_lambda,
            'dens_frac': self.dens_frac,
            'dens_var_shift': self.dens_var_shift,
            'output_dens': self.output_dens,
            'disconnection_distance': self.disconnection_distance,
            'precomputed_knn': self.precomputed_knn,
            'backend': self.backend,
            'prefer_gpu': self.prefer_gpu
        }
        return params
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters
            
        Returns
        -------
        self : object
            Returns self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Reset estimator when parameters change
        self._estimator = None
        self._chosen_backend = None
        
        return self
    
    def __repr__(self):
        """String representation of the UMAP object."""
        backend_str = f", backend={self._chosen_backend}" if self._chosen_backend else ""
        return f"UMAP(n_components={self.n_components}, n_neighbors={self.n_neighbors}{backend_str})"