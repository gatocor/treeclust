"""
Sklearn-style TSNE implementation with backend dispatch.

This module provides a sklearn-compatible TSNE class that can dispatch to
different backends while maintaining a consistent API.
"""

import numpy as np
import warnings
from typing import Optional, Union, Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE as SklearnTSNE

# Import centralized availability flags
from .. import CUML_AVAILABLE

# Import cuML backend conditionally
if CUML_AVAILABLE:
    try:
        from cuml.manifold import TSNE as CumlTSNE
    except ImportError:
        CUML_AVAILABLE = False

class TSNE(BaseEstimator, TransformerMixin):
    """
    t-SNE (t-Distributed Stochastic Neighbor Embedding) with backend dispatch.
    
    This class provides a sklearn-compatible interface for t-SNE that can
    automatically choose between different backends (sklearn, cuML) based
    on availability and user preferences.
    
    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms
    learning_rate : float or 'warn', default='warn'
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]
    n_iter : int, default=1000
        Maximum number of iterations for the optimization
    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before stopping
    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will be stopped
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances
    init : str, default='random'
        Initialization of embedding
    verbose : int, default=0
        Verbosity level
    random_state : int, optional
        Random seed for reproducible results
    method : str, default='barnes_hut'
        Algorithm to use ('barnes_hut' or 'exact')
    angle : float, default=0.5
        Only used if method='barnes_hut'
    n_jobs : int, optional
        Number of parallel jobs (sklearn only)
    backend : str, optional
        Backend to use ('sklearn', 'cuml', 'auto')
    prefer_gpu : bool, default=False
        Whether to prefer GPU implementations when available
    
    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the embedding vectors
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization
    n_features_in_ : int
        Number of features seen during fit
    feature_names_in_ : ndarray of shape (n_features_in_,), optional
        Names of features seen during fit
    """
    
    def __init__(
        self,
        n_components: int = 2,
        *,
        perplexity: float = 30.0,
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
        backend: Optional[str] = None,
        prefer_gpu: bool = False
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.backend = backend
        self.prefer_gpu = prefer_gpu
        
        # Initialize the backend estimator
        self._estimator = None
        self._chosen_backend = None
    
    def _choose_backend(self):
        """Choose the appropriate backend based on preferences and availability."""
        if self.backend is not None:
            # Explicit backend requested
            if self.backend == 'sklearn':
                return 'sklearn'
            elif self.backend == 'cuml' and not CUML_AVAILABLE:
                raise ImportError("cuml backend requested but not available")
            return self.backend
        
        # Auto-select backend
        if self.prefer_gpu and CUML_AVAILABLE:
            return 'cuml'
        else:
            return 'sklearn'
    
    def _create_estimator(self):
        """Create the appropriate backend estimator."""
        self._chosen_backend = self._choose_backend()
        
        if self._chosen_backend == 'sklearn':
            # Create sklearn t-SNE
            sklearn_params = {
                'n_components': self.n_components,
                'perplexity': self.perplexity,
                'learning_rate': self.learning_rate,
                'n_iter': self.n_iter,
                'n_iter_without_progress': self.n_iter_without_progress,
                'min_grad_norm': self.min_grad_norm,
                'metric': self.metric,
                'init': self.init,
                'verbose': self.verbose,
                'random_state': self.random_state,
                'method': self.method,
                'angle': self.angle
            }
            if self.n_jobs is not None:
                sklearn_params['n_jobs'] = self.n_jobs
            
            self._estimator = SklearnTSNE(**sklearn_params)
            
        elif self._chosen_backend == 'cuml':
            # Create cuML t-SNE
            cuml_params = {
                'n_components': self.n_components,
                'perplexity': self.perplexity,
                'learning_rate': self.learning_rate,
                'n_iter': self.n_iter,
                'min_grad_norm': self.min_grad_norm,
                'metric': self.metric,
                'init': self.init,
                'verbose': self.verbose,
                'random_state': self.random_state,
                'method': self.method,
                'angle': self.angle
            }
            # Note: cuML may not support all parameters
            self._estimator = CumlTSNE(**{k: v for k, v in cuml_params.items() 
                                       if k in CumlTSNE.__init__.__code__.co_varnames})
    
    def fit(self, X, y=None):
        """
        Fit the model with X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency
            
        Returns
        -------
        self : object
            Returns self
        """
        if self._estimator is None:
            self._create_estimator()
        
        # Fit the estimator
        self._estimator.fit(X)
        
        # Copy attributes from the backend estimator
        if hasattr(self._estimator, 'embedding_'):
            self.embedding_ = self._estimator.embedding_
        if hasattr(self._estimator, 'kl_divergence_'):
            self.kl_divergence_ = self._estimator.kl_divergence_
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
        y : Ignored
            Not used, present for API consistency
            
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space
        """
        if self._estimator is None:
            self._create_estimator()
        
        # Use fit_transform from backend
        X_transformed = self._estimator.fit_transform(X)
        
        # Copy attributes from the backend estimator
        if hasattr(self._estimator, 'embedding_'):
            self.embedding_ = self._estimator.embedding_
        if hasattr(self._estimator, 'kl_divergence_'):
            self.kl_divergence_ = self._estimator.kl_divergence_
        if hasattr(self._estimator, 'n_features_in_'):
            self.n_features_in_ = self._estimator.n_features_in_
        if hasattr(self._estimator, 'feature_names_in_'):
            self.feature_names_in_ = self._estimator.feature_names_in_
        
        return X_transformed
    
    def transform(self, X):
        """
        Transform X to the embedded space.
        
        Note: t-SNE does not support transform after fit.
        This method raises NotImplementedError.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Raises
        ------
        NotImplementedError
            t-SNE does not support transform after fit
        """
        raise NotImplementedError("t-SNE does not support transform after fit. Use fit_transform instead.")
    
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
            'perplexity': self.perplexity,
            'learning_rate': self.learning_rate,
            'n_iter': self.n_iter,
            'n_iter_without_progress': self.n_iter_without_progress,
            'min_grad_norm': self.min_grad_norm,
            'metric': self.metric,
            'init': self.init,
            'verbose': self.verbose,
            'random_state': self.random_state,
            'method': self.method,
            'angle': self.angle,
            'n_jobs': self.n_jobs,
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
        """String representation of the TSNE object."""
        backend_str = f", backend={self._chosen_backend}" if self._chosen_backend else ""
        return f"TSNE(n_components={self.n_components}, perplexity={self.perplexity}{backend_str})"