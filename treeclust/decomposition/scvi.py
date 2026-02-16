from typing import Optional, Dict, Any, Union
import warnings

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False
    warnings.warn("AnnData not available. ScviWrapper will have limited functionality.")

class ScviWrapper:
    """
    sklearn-style wrapper for scvi-tools models.
    
    This class wraps any scvi-tools model to provide sklearn-compatible
    fit, transform, and fit_transform methods. It automatically checks
    for required methods (train, setup_anndata, get_latent_representation)
    and provides a clean interface.
    
    Parameters
    ----------
    scvi_class : class
        The scvi-tools model class (e.g., scvi.model.SCVI, scvi.model.scANVI)
    setup_kwargs : dict, optional
        Keyword arguments to pass to setup_anndata method
    model_kwargs : dict, optional
        Keyword arguments to pass to model constructor
    train_kwargs : dict, optional
        Keyword arguments to pass to train method
    
    Attributes
    ----------
    scvi_class_ : class
        The wrapped scvi-tools model class
    model_ : object
        The fitted scvi-tools model instance
    is_fitted_ : bool
        Whether the model has been fitted
    setup_kwargs_ : dict
        Stored setup arguments
    model_kwargs_ : dict
        Stored model arguments
    train_kwargs_ : dict
        Stored training arguments
    
    Examples
    --------
    >>> import numpy as np
    >>> import scvi
    >>> from treeclust.dimensionality_reduction import ScviWrapper
    >>> 
    >>> # Create count matrix (cells x genes)
    >>> X = np.random.negative_binomial(5, 0.3, size=(1000, 2000))
    >>> batch_labels = np.random.choice(['batch1', 'batch2'], size=1000)
    >>> 
    >>> # Wrap SCVI model
    >>> wrapper = ScviWrapper(
    ...     scvi_class=scvi.model.SCVI,
    ...     setup_kwargs={'batch_key': 'batch'},
    ...     train_kwargs={'max_epochs': 100}
    ... )
    >>> 
    >>> # Use sklearn-style API with matrix input
    >>> X_latent = wrapper.fit_transform(X, obs_data={'batch': batch_labels})
    >>> 
    >>> # Or separate fit and transform
    >>> wrapper.fit(X, obs_data={'batch': batch_labels})
    >>> X_latent = wrapper.transform(X)
    """
    
    def __init__(
        self,
        scvi_class,
        setup_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        train_kwargs: Optional[dict] = None
    ):
        """
        Initialize the scvi-tools wrapper.
        
        Parameters
        ----------
        scvi_class : class
            The scvi-tools model class to wrap
        setup_kwargs : dict, optional
            Arguments for setup_anndata method
        model_kwargs : dict, optional
            Arguments for model constructor
        train_kwargs : dict, optional
            Arguments for train method
        """
        # Validate that the class has required methods
        self._validate_scvi_class(scvi_class)
        
        self.scvi_class_ = scvi_class
        self.setup_kwargs_ = setup_kwargs or {}
        self.model_kwargs_ = model_kwargs or {}
        self.train_kwargs_ = train_kwargs or {}
        
        # State attributes
        self.model_ = None
        self.is_fitted_ = False
        self.adata_setup_ = None
    
    def _validate_scvi_class(self, scvi_class):
        """
        Validate that the scvi class has required methods.
        
        Parameters
        ----------
        scvi_class : class
            The scvi-tools model class to validate
        
        Raises
        ------
        ValueError
            If required methods are missing
        """
        required_methods = ['setup_anndata', 'train', 'get_latent_representation']
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(scvi_class, method):
                missing_methods.append(method)
        
        if missing_methods:
            raise ValueError(
                f"scvi class {scvi_class.__name__} is missing required methods: "
                f"{missing_methods}. Required methods are: {required_methods}"
            )
    
    def _create_anndata(self, X, obs_data=None):
        """
        Create AnnData object from matrix input.
        
        Parameters
        ----------
        X : array-like
            Data matrix (cells x genes)
        obs_data : dict, optional
            Observation metadata to include
        
        Returns
        -------
        adata : AnnData
            Created AnnData object
        """
        try:
            import anndata as ad
        except ImportError:
            raise ImportError("anndata is required for ScviWrapper. Install with: pip install anndata")
        
        # Convert to numpy array if needed
        import numpy as np
        if hasattr(X, 'toarray'):  # sparse matrix
            X = X.toarray()
        X = np.asarray(X)
        
        # Create AnnData object
        adata = ad.AnnData(X)
        
        # Add observation data if provided
        if obs_data is not None:
            for key, values in obs_data.items():
                adata.obs[key] = values
        
        # Generate default cell and gene names
        adata.obs_names = [f'cell_{i}' for i in range(adata.n_obs)]
        adata.var_names = [f'gene_{i}' for i in range(adata.n_vars)]
        
        return adata
    
    def fit(self, X, y=None, obs_data=None):
        """
        Fit the scvi-tools model to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_cells, n_genes)
            Data matrix to fit the model on
        y : ignored
            Not used, present for sklearn compatibility
        obs_data : dict, optional
            Dictionary of observation metadata (e.g., {'batch': batch_labels})
        
        Returns
        -------
        self : ScviWrapper
            Returns self for method chaining
        """
        # Create AnnData object from matrix input
        if hasattr(X, 'n_obs'):  # Already an AnnData object
            adata = X.copy()
        else:
            adata = self._create_anndata(X, obs_data)
        
        # Store the adata for potential reuse
        self.adata_setup_ = adata.copy()
        
        # Setup the adata
        try:
            self.scvi_class_.setup_anndata(adata, **self.setup_kwargs_)
        except Exception as e:
            raise RuntimeError(f"Failed to setup AnnData: {str(e)}")
        
        # Create model instance
        try:
            self.model_ = self.scvi_class_(adata, **self.model_kwargs_)
        except Exception as e:
            raise RuntimeError(f"Failed to create model: {str(e)}")
        
        # Train the model
        try:
            self.model_.train(**self.train_kwargs_)
        except Exception as e:
            raise RuntimeError(f"Failed to train model: {str(e)}")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X=None, obs_data=None):
        """
        Get latent representation from the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_cells, n_genes), optional
            Data matrix to transform. If None, uses the data used for fitting.
        obs_data : dict, optional
            Dictionary of observation metadata for new data
        
        Returns
        -------
        X_latent : numpy.ndarray
            Latent representation of the data
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before transform. Call fit() first.")
        
        # Use fitted data if no X provided
        if X is None:
            if self.adata_setup_ is None:
                raise RuntimeError("No data available for transform. Provide X argument.")
            adata = self.adata_setup_
        else:
            # Create AnnData object from matrix input
            if hasattr(X, 'n_obs'):  # Already an AnnData object
                adata = X
            else:
                adata = self._create_anndata(X, obs_data)
        
        # Get latent representation
        try:
            X_latent = self.model_.get_latent_representation(adata)
        except Exception as e:
            raise RuntimeError(f"Failed to get latent representation: {str(e)}")
        
        return X_latent
    
    def fit_transform(self, X, y=None, obs_data=None):
        """
        Fit the model and return the latent representation.
        
        Parameters
        ----------
        X : array-like of shape (n_cells, n_genes)
            Data matrix to fit and transform
        y : ignored
            Not used, present for sklearn compatibility
        obs_data : dict, optional
            Dictionary of observation metadata
        
        Returns
        -------
        X_latent : numpy.ndarray
            Latent representation of the data
        """
        return self.fit(X, y, obs_data).transform(X, obs_data)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, optional
            If True, return parameters for this estimator and
            contained subobjects
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values
        """
        return {
            'scvi_class': self.scvi_class_,
            'setup_kwargs': self.setup_kwargs_,
            'model_kwargs': self.model_kwargs_,
            'train_kwargs': self.train_kwargs_
        }
    
    def set_params(self, **params):
        """
        Set parameters for this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters
        
        Returns
        -------
        self : ScviWrapper
            Returns self for method chaining
        """
        for key, value in params.items():
            if key == 'scvi_class':
                self._validate_scvi_class(value)
                self.scvi_class_ = value
            elif key == 'setup_kwargs':
                self.setup_kwargs_ = value or {}
            elif key == 'model_kwargs':
                self.model_kwargs_ = value or {}
            elif key == 'train_kwargs':
                self.train_kwargs_ = value or {}
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        # Reset fitted state if parameters change
        self.is_fitted_ = False
        self.model_ = None
        return self
    
    def get_model(self):
        """
        Get the underlying scvi-tools model.
        
        Returns
        -------
        model : object
            The fitted scvi-tools model instance
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first. Call fit() method.")
        return self.model_
    
    def save(self, dir_path, overwrite=False, save_anndata=False, **kwargs):
        """
        Save the fitted model.
        
        Parameters
        ----------
        dir_path : str
            Path to save the model
        overwrite : bool, optional
            Whether to overwrite existing files
        save_anndata : bool, optional
            Whether to save the anndata object
        **kwargs
            Additional arguments passed to model.save()
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before saving. Call fit() first.")
        
        self.model_.save(dir_path, overwrite=overwrite, save_anndata=save_anndata, **kwargs)
    
    @classmethod
    def load(cls, dir_path, adata=None, **kwargs):
        """
        Load a saved model.
        
        Parameters
        ----------
        dir_path : str
            Path to the saved model
        adata : AnnData, optional
            Annotated data object to associate with the model
        **kwargs
            Additional arguments passed to model.load()
        
        Returns
        -------
        wrapper : ScviWrapper
            Loaded wrapper instance
        """
        # This is a simplified load - in practice, you'd need to
        # determine the original scvi_class and parameters
        raise NotImplementedError(
            "Loading is not fully implemented. "
            "Use the original scvi-tools load method and wrap the result."
        )
    
    def __repr__(self):
        """String representation of the wrapper."""
        return (f"ScviWrapper(scvi_class={self.scvi_class_.__name__ if self.scvi_class_ else None}, "
                f"fitted={self.is_fitted_})")