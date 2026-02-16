"""
Common pipelines for treeclust.

This module provides simple functions that return sklearn Pipeline objects 
combining matrix bootstrapping with dimensionality reduction techniques.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
import warnings

# Import sklearn components (required dependency)
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.decomposition import PCA

# Import centralized availability flags
from .. import DIM_REDUCTION_AVAILABLE

__all__ = [
    'PipelineBootstrapper',
    'pipeline_matrix_bootstrap_pca', 
    'pipeline_matrix_bootstrap_vae',
    'MultiPipelineBootstrapper',
    'BootstrapTransformIterator',
    'BootstrapPredictIterator'
]


class BootstrapTransformIterator:
    """
    Iterator for bootstrap transform results that computes transformations on-demand.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        partitions: List[Dict[str, np.ndarray]],
        fitted_models: List[Any],
        cache_results: bool = False
    ):
        self.X = X
        self.y = y
        self.partitions = partitions
        self.fitted_models = fitted_models
        self.cache_results = cache_results
        self.cached_results = {} if cache_results else None
        self.index = 0
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.partitions):
            raise StopIteration
        
        # Check cache first
        if self.cache_results and self.index in self.cached_results:
            result = self.cached_results[self.index]
            self.index += 1
            return result
        
        # Compute transformation on-demand
        split_info = self.partitions[self.index]
        fitted_model = self.fitted_models[self.index]
        
        obs_train_idx = split_info['obs_train_idx']
        feat_train_idx = split_info['feat_train_idx']
        X_split = self.X[np.ix_(obs_train_idx, feat_train_idx)]
        
        # Handle empty pipeline case (no transformations)
        if fitted_model is None:
            X_transformed = X_split
        else:
            X_transformed = fitted_model.transform(X_split)
        
        # Cache if requested
        if self.cache_results:
            self.cached_results[self.index] = X_transformed
        
        self.index += 1
        return X_transformed
    
    def __len__(self):
        return len(self.partitions)
    
    def __getitem__(self, index):
        """Allow direct indexing access."""
        if index >= len(self.partitions):
            raise IndexError("Index out of range")
        
        # Check cache first
        if self.cache_results and index in self.cached_results:
            return self.cached_results[index]
        
        # Compute transformation for specific index
        split_info = self.partitions[index]
        fitted_model = self.fitted_models[index]
        
        obs_train_idx = split_info['obs_train_idx']
        feat_train_idx = split_info['feat_train_idx']
        X_split = self.X[np.ix_(obs_train_idx, feat_train_idx)]
        
        # Handle empty pipeline case (no transformations)
        if fitted_model is None:
            X_transformed = X_split
        else:
            X_transformed = fitted_model.transform(X_split)
        
        # Cache if requested
        if self.cache_results:
            self.cached_results[index] = X_transformed
        
        return X_transformed


class BootstrapPredictIterator:
    """
    Iterator for bootstrap prediction results that computes predictions on-demand.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        partitions: List[Dict[str, np.ndarray]],
        fitted_models: List[Any],
        cache_results: bool = False
    ):
        self.X = X
        self.y = y
        self.partitions = partitions
        self.fitted_models = fitted_models
        self.cache_results = cache_results
        self.cached_results = {} if cache_results else None
        self.index = 0
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.partitions):
            raise StopIteration
        
        # Check cache first
        if self.cache_results and self.index in self.cached_results:
            result = self.cached_results[self.index]
            self.index += 1
            return result
        
        # Compute prediction on-demand
        split_info = self.partitions[self.index]
        fitted_model = self.fitted_models[self.index]
        
        obs_test_idx = split_info['obs_test_idx']
        feat_train_idx = split_info['feat_train_idx']
        
        if len(obs_test_idx) > 0:
            X_test = self.X[np.ix_(obs_test_idx, feat_train_idx)]
            # Handle empty pipeline case (no transformations)
            if fitted_model is None:
                y_pred = np.array([])  # Can't predict without a model
            else:
                y_pred = fitted_model.predict(X_test)
        else:
            y_pred = np.array([])
        
        # Cache if requested
        if self.cache_results:
            self.cached_results[self.index] = y_pred
        
        self.index += 1
        return y_pred
    
    def __len__(self):
        return len(self.partitions)
    
    def __getitem__(self, index):
        """Allow direct indexing access."""
        if index >= len(self.partitions):
            raise IndexError("Index out of range")
        
        # Check cache first
        if self.cache_results and index in self.cached_results:
            return self.cached_results[index]
        
        # Compute prediction for specific index
        split_info = self.partitions[index]
        fitted_model = self.fitted_models[index]
        
        obs_test_idx = split_info['obs_test_idx']
        feat_train_idx = split_info['feat_train_idx']
        
        if len(obs_test_idx) > 0:
            X_test = self.X[np.ix_(obs_test_idx, feat_train_idx)]
            # Handle empty pipeline case (no transformations)
            if fitted_model is None:
                y_pred = np.array([])  # Can't predict without a model
            else:
                y_pred = fitted_model.predict(X_test)
        else:
            y_pred = np.array([])
        
        # Cache if requested
        if self.cache_results:
            self.cached_results[index] = y_pred
        
        return y_pred

# Import optional decomposition components conditionally
if DIM_REDUCTION_AVAILABLE:
    try:
        from ..decomposition.vae import VAE
        from ..decomposition.scvi import ScviWrapper
    except ImportError:
        pass

class PipelineBootstrapper(Pipeline):
    """
    Pipeline class with configurable bootstrapping step at the beginning.
    
    This is a specialized Pipeline that accepts observation and feature bootstrapping
    making it easy to combine different bootstrapping strategies with other transforms.
    
    Features:
    - Accepts observation bootstrapper (e.g., KFold) 
    - Accepts feature splitter (e.g., KFold) for feature selection
    - Compatible with sklearn Pipeline interface
    - Easy access to bootstrap split results
    - Supports all standard Pipeline functionality
    """
    
    def __init__(
        self,
        steps: List[Tuple[str, Any]] = [],
        observation_splitter: Optional[Any] = ShuffleSplit(),
        feature_splitter: Optional[Any] = None,
        random_state: Optional[int] = None,
        n_splits: int = 10,
        memory: Optional[Any] = None,
        cache_results: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the PipelineBootstrapper.
        
        Parameters
        ----------
        steps : list of tuples
            List of (name, estimator) tuples defining the pipeline steps.
            The bootstrap transformer will be automatically added as the first step.
        observation_splitter : Any, optional
            Bootstrapper for observations/rows (e.g., KFold).
            If None, no observation bootstrapping is applied.
        feature_splitter : Any, optional  
            Cross-validation splitter for features/columns (e.g., KFold).
            If None, no feature bootstrapping is applied.
        random_state : int, optional
            Random seed for reproducible results
        n_splits : int, default=10
            Number of splits for the bootstrapping
        memory : object, optional
            Caching object (sklearn Pipeline parameter)
        cache_results : bool, default=False
            If True, cache all transformation results in memory for faster access.
            If False, use iterators that compute results on-demand (memory efficient).
        verbose : bool, default=False
            Whether to print progress information
        """
        # Store all constructor parameters as attributes (required for sklearn clone)
        self.observation_splitter = observation_splitter
        self.feature_splitter = feature_splitter
        self.random_state = random_state
        self.n_splits = n_splits
        self.cache_results = cache_results
        
        # Configure splitters with the provided parameters
        if self.observation_splitter is not None:
            if hasattr(self.observation_splitter, 'random_state'):
                self.observation_splitter.random_state = random_state
            if hasattr(self.observation_splitter, 'n_splits'):
                self.observation_splitter.n_splits = n_splits

        if self.feature_splitter is not None:
            if hasattr(self.feature_splitter, 'random_state'):
                self.feature_splitter.random_state = random_state
            if hasattr(self.feature_splitter, 'n_splits'):
                self.feature_splitter.n_splits = n_splits
        
        # Initialize parent Pipeline
        super().__init__(steps=steps, memory=memory, verbose=verbose)
        
        # Initialize storage for bootstrap partitions
        self.bootstrap_partitions_ = []
        self.fitted_models_ = []
        self.cached_transforms_ = {}  # Only used when cache_results=True
        self.cached_predictions_ = {}  # Only used when cache_results=True
        self.is_fitted_ = False
    
    def set_n_splits(self, n_splits: int):
        """
        Set the number of splits for both observation and feature splitters.
        
        This method propagates the n_splits parameter to the underlying splitters
        that support it (like KFold, ShuffleSplit, etc.).
        
        Parameters
        ----------
        n_splits : int
            Number of splits to set for the splitters
        """
        # Update the stored parameter
        self.n_splits = n_splits
        
        # Propagate to observation splitter if it supports n_splits
        if self.observation_splitter is not None and hasattr(self.observation_splitter, 'n_splits'):
            self.observation_splitter.n_splits = n_splits
            
        # Propagate to feature splitter if it supports n_splits  
        if self.feature_splitter is not None and hasattr(self.feature_splitter, 'n_splits'):
            self.feature_splitter.n_splits = n_splits
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate bootstrap splits for both observations and features.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray, optional
            Target values
            
        Returns
        -------
        splits : List[Tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) tuples for each bootstrap split
        """
        splits = []
        
        # Generate observation splits
        if self.observation_splitter is not None:
            obs_splits = list(self.observation_splitter.split(X, y))
        else:
            # Default to using all data
            obs_splits = [(np.arange(X.shape[0]), np.array([]))]
        
        # For each observation split, also apply feature splitting if specified
        for obs_train_idx, obs_test_idx in obs_splits:
            if self.feature_splitter is not None:
                # Apply feature splitting on the training data
                X_train_obs = X[obs_train_idx]
                feat_splits = list(self.feature_splitter.split(X_train_obs.T))
                
                for feat_train_idx, feat_test_idx in feat_splits:
                    # Store both observation and feature indices
                    split_info = {
                        'obs_train_idx': obs_train_idx,
                        'obs_test_idx': obs_test_idx,
                        'feat_train_idx': feat_train_idx,
                        'feat_test_idx': feat_test_idx
                    }
                    splits.append((split_info, None))  # Second element for compatibility
            else:
                # Only observation splitting
                split_info = {
                    'obs_train_idx': obs_train_idx,
                    'obs_test_idx': obs_test_idx,
                    'feat_train_idx': np.arange(X.shape[1]),
                    'feat_test_idx': np.array([])
                }
                splits.append((split_info, None))
        
        return splits

    def split_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PipelineBootstrapper':
        """
        Fit the pipeline on multiple bootstrap splits and store all fitted models.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray, optional
            Target values
            
        Returns
        -------
        self : PipelineBootstrapper
            Fitted pipeline with stored partitions and models
        """
        # Clear previous fits
        self.bootstrap_partitions_ = []
        self.fitted_models_ = []
        
        # Get all bootstrap splits
        splits = self.split(X, y)
        
        for split_info, _ in splits:
            # Extract data for this split
            obs_train_idx = split_info['obs_train_idx']
            feat_train_idx = split_info['feat_train_idx']
            
            # Get training data for this bootstrap split
            X_train = X[np.ix_(obs_train_idx, feat_train_idx)]
            y_train = y[obs_train_idx] if y is not None else None
            
            # Handle empty pipeline case (just splitting, no transformations)
            if len(self.steps) == 0:
                # For empty pipeline, we just store the split info without fitting anything
                fitted_model = None
            else:
                # Clone the pipeline and fit on this bootstrap split
                pipeline_clone = clone(self)
                pipeline_clone.fit(X_train, y_train)
                fitted_model = pipeline_clone
            
            # Store the partition info and fitted model
            self.bootstrap_partitions_.append(split_info)
            self.fitted_models_.append(fitted_model)
        
        self.is_fitted_ = True
        return self
    
    def split_fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Union[BootstrapTransformIterator, List[np.ndarray]]:
        """
        Fit and transform the data on all bootstrap splits.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray, optional
            Target values
            
        Returns
        -------
        transformed_data : BootstrapTransformIterator or List[np.ndarray]
            Iterator over transformed data for each bootstrap split (memory efficient)
            or List of arrays if cache_results=True
        """
        # First fit on all splits
        self.split_fit(X, y)
        
        # Return iterator for memory efficiency
        transform_iterator = BootstrapTransformIterator(
            X=X,
            y=y,
            partitions=self.bootstrap_partitions_,
            fitted_models=self.fitted_models_,
            cache_results=self.cache_results
        )
        
        if self.cache_results:
            # If caching is requested, compute all results and return as list
            return list(transform_iterator)
        else:
            # Return iterator for memory efficiency
            return transform_iterator
        
    def split_fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Union[BootstrapPredictIterator, List[np.ndarray]]:
        """
        Fit and predict on all bootstrap splits.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray, optional
            Target values
            
        Returns
        -------
        predictions : BootstrapPredictIterator or List[np.ndarray]
            Iterator over predictions for each bootstrap split (memory efficient)
            or List of arrays if cache_results=True
        """
        # First fit on all splits
        self.split_fit(X, y)
        
        # Return iterator for memory efficiency
        predict_iterator = BootstrapPredictIterator(
            X=X,
            y=y,
            partitions=self.bootstrap_partitions_,
            fitted_models=self.fitted_models_,
            cache_results=self.cache_results
        )
        
        if self.cache_results:
            # If caching is requested, compute all results and return as list
            return list(predict_iterator)
        else:
            # Return iterator for memory efficiency
            return predict_iterator
    
    def get_bootstrap_partitions(self) -> List[Dict[str, np.ndarray]]:
        """
        Get the stored bootstrap partitions.
        
        Returns
        -------
        partitions : List[Dict[str, np.ndarray]]
            List of partition dictionaries containing train/test indices for observations and features
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted first to get bootstrap partitions.")
        return self.bootstrap_partitions_
    
    def get_fitted_models(self) -> List[Any]:
        """
        Get the fitted models for each bootstrap partition.
        
        Returns
        -------
        models : List[Any]
            List of fitted pipeline models
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted first to get fitted models.")
        return self.fitted_models_

def pipeline_matrix_bootstrap_pca(
    sample_ratio: float = 0.8,
    feature_ratio: Optional[float] = None,
    n_components: int = 50,
    whiten: bool = False,
    cache_results: bool = False,
    random_state: Optional[int] = None
) -> 'PipelineBootstrapper':
    """
    Create a matrix bootstrap + PCA pipeline.
    
    Parameters
    ----------
    sample_ratio : float, default=0.8
        Ratio of samples to include in bootstrap
    feature_ratio : float, optional
        Ratio of features to include in bootstrap
    n_components : int, default=50
        Number of PCA components
    whiten : bool, default=False
        Whether to whiten the components
    cache_results : bool, default=False
        If True, cache transformation results in memory.
        If False, use iterators (memory efficient).
    random_state : int, optional
        Random seed
        
    Returns
    -------
    pipeline : PipelineBootstrapper
        Pipeline with bootstrap splitting and PCA
    """

    
    # Create observation splitter
    observation_splitter = ShuffleSplit(
        n_splits=10,
        train_size=sample_ratio,
        random_state=random_state
    )
    
    # Create feature splitter if specified
    feature_splitter = None
    if feature_ratio is not None:
        feature_splitter = ShuffleSplit(
            n_splits=5,
            train_size=feature_ratio,
            random_state=random_state
        )
        
    return PipelineBootstrapper(
        ('pca', PCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state
        )),
        observation_splitter=observation_splitter,
        feature_splitter=feature_splitter,
        cache_results=cache_results,
        random_state=random_state
    )

def pipeline_matrix_bootstrap_vae(
    sample_ratio: float = 0.8,
    feature_ratio: Optional[float] = None,
    latent_dim: int = 10,
    encoder_layers: Optional[List[int]] = None,
    decoder_layers: Optional[List[int]] = None,
    activation: str = 'relu',
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 100,
    cache_results: bool = False,
    random_state: Optional[int] = None
) -> 'PipelineBootstrapper':
    """
    Create a matrix bootstrap + VAE pipeline.
    
    Parameters
    ----------
    sample_ratio : float, default=0.8
        Ratio of samples to include in bootstrap
    feature_ratio : float, optional
        Ratio of features to include in bootstrap
    latent_dim : int, default=10
        VAE latent dimension
    encoder_layers : List[int], optional
        VAE encoder layer sizes
    decoder_layers : List[int], optional
        VAE decoder layer sizes
    activation : str, default='relu'
        Activation function
    learning_rate : float, default=1e-3
        Learning rate for training
    batch_size : int, default=64
        Batch size for training
    epochs : int, default=100
        Number of training epochs
    cache_results : bool, default=False
        If True, cache transformation results in memory.
        If False, use iterators (memory efficient).
    random_state : int, optional
        Random seed
        
    Returns
    -------
    pipeline : PipelineBootstrapper
        Pipeline with bootstrap splitting and VAE
    """

    
    # Create observation splitter
    observation_splitter = ShuffleSplit(
        n_splits=10,
        train_size=sample_ratio,
        random_state=random_state
    )
    
    # Create feature splitter if specified
    feature_splitter = None
    if feature_ratio is not None:
        feature_splitter = ShuffleSplit(
            n_splits=5,
            train_size=feature_ratio,
            random_state=random_state
        )
        
    return PipelineBootstrapper(
        ('vae', VAE(
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activation=activation,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            random_state=random_state
        )),
        observation_splitter=observation_splitter,
        feature_splitter=feature_splitter,
        cache_results=cache_results,
        random_state=random_state
    )


class MultiPipelineBootstrapper:
    """
    Manager class that can take several pipelines and distribute samples between them.
    
    This class takes multiple (name, pipeline) tuples and distributes a total number 
    of samples between the pipelines, allowing for ensemble sampling from different
    pipeline configurations.
    """
    
    def __init__(
        self,
        *pipeline_tuples: Tuple[str, PipelineBootstrapper],
        n_samples: int = 100,
        random_state: Optional[int] = None
    ):
        """
        Initialize the MultiPipelineBootstrapper.
        
        Parameters
        ----------
        *pipeline_tuples : Tuple[str, PipelineBootstrapper]
            Variable number of (name, pipeline) tuples
        n_samples : int, default=100
            Total number of samples to distribute between all pipelines
        random_state : int, optional
            Random seed for reproducible results
        """
        if not pipeline_tuples:
            raise ValueError("At least one pipeline tuple must be provided")
        
        self.pipeline_tuples = pipeline_tuples
        self.pipeline_names = [name for name, _ in pipeline_tuples]
        self.pipelines = [pipeline for _, pipeline in pipeline_tuples]
        self.n_pipelines = len(pipeline_tuples)
        self.n_samples = n_samples
        self.random_state = random_state
        
        # Distribute samples between pipelines
        samples_per_pipeline = n_samples // self.n_pipelines
        extra_samples = n_samples % self.n_pipelines
        
        self.samples_distribution = []
        for i in range(self.n_pipelines):
            n_samples_pipeline = samples_per_pipeline + (1 if i < extra_samples else 0)
            self.samples_distribution.append(n_samples_pipeline)
        
        # Set random states for pipelines if specified
        if random_state is not None:
            np.random.seed(random_state)
            for i, (_, pipeline) in enumerate(pipeline_tuples):
                pipeline_seed = random_state + i
                if hasattr(pipeline, 'random_state'):
                    pipeline.random_state = pipeline_seed
                if hasattr(pipeline.observation_splitter, 'random_state'):
                    pipeline.observation_splitter.random_state = pipeline_seed
                if hasattr(pipeline.feature_splitter, 'random_state') and pipeline.feature_splitter is not None:
                    pipeline.feature_splitter.random_state = pipeline_seed + 1000
        
        # Results storage
        self.is_fitted_ = False
        self.sample_results_ = []
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MultiPipelineBootstrapper':
        """
        Fit all pipelines and generate distributed samples.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
        y : np.ndarray, optional
            Target values of shape (n_samples,)
            
        Returns
        -------
        self : MultiPipelineBootstrapper
            Fitted multi-pipeline manager with samples
        """
        self.sample_results_ = []
        
        # Fit each pipeline and collect samples according to distribution
        for i, ((name, pipeline), n_samples_pipeline) in enumerate(zip(self.pipeline_tuples, self.samples_distribution)):
            # Fit the pipeline
            pipeline.split_fit(X, y)
            
            # Get bootstrap partitions and fitted models
            partitions = pipeline.get_bootstrap_partitions()
            fitted_models = pipeline.get_fitted_models()
            
            # Sample from this pipeline
            pipeline_samples = []
            for sample_idx in range(n_samples_pipeline):
                # Randomly select a partition/model pair
                partition_idx = np.random.randint(len(partitions))
                partition = partitions[partition_idx]
                fitted_model = fitted_models[partition_idx]
                
                # Extract data for this partition
                obs_train_idx = partition['obs_train_idx']
                feat_train_idx = partition['feat_train_idx']
                X_sample = X[np.ix_(obs_train_idx, feat_train_idx)]
                y_sample = y[obs_train_idx] if y is not None else None
                
                sample_info = {
                    'pipeline_name': name,
                    'sample_idx': sample_idx,
                    'partition_idx': partition_idx,
                    'X_sample': X_sample,
                    'y_sample': y_sample,
                    'fitted_model': fitted_model,
                    'partition': partition
                }
                pipeline_samples.append(sample_info)
            
            self.sample_results_.extend(pipeline_samples)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, apply_transform: bool = True) -> List[np.ndarray]:
        """
        Transform the sampled data using their corresponding fitted models.
        
        Parameters
        ----------
        apply_transform : bool, default=True
            Whether to apply transformation to the samples
            
        Returns
        -------
        transformed_samples : List[np.ndarray]
            List of transformed samples from all pipelines
        """
        if not self.is_fitted_:
            raise ValueError("MultiPipelineBootstrapper must be fitted before transform")
        
        transformed_samples = []
        
        for sample_info in self.sample_results_:
            if apply_transform:
                X_transformed = sample_info['fitted_model'].transform(sample_info['X_sample'])
                transformed_samples.append(X_transformed)
            else:
                transformed_samples.append(sample_info['X_sample'])
        
        return transformed_samples
    
    def predict(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Predict using all sampled models.
        
        Parameters
        ----------
        X : np.ndarray
            Input data to predict on
            
        Returns
        -------
        predictions : List[np.ndarray]
            List of predictions from all sampled models
        """
        if not self.is_fitted_:
            raise ValueError("MultiPipelineBootstrapper must be fitted before predict")
        
        predictions = []
        
        for sample_info in self.sample_results_:
            # Apply same feature selection as training
            feat_idx = sample_info['partition']['feat_train_idx']
            X_feat = X[:, feat_idx]
            prediction = sample_info['fitted_model'].predict(X_feat)
            predictions.append(prediction)
        
        return predictions
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """
        Get all sample information.
        
        Returns
        -------
        samples : List[Dict[str, Any]]
            List of sample dictionaries containing all sample information
        """
        if not self.is_fitted_:
            raise ValueError("MultiPipelineBootstrapper must be fitted to get samples")
        return self.sample_results_
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the multi-pipeline configuration.
        
        Returns
        -------
        summary : Dict[str, Any]
            Summary information about the configuration
        """
        summary = {
            'n_pipelines': self.n_pipelines,
            'n_samples': self.n_samples,
            'pipeline_names': self.pipeline_names,
            'samples_distribution': dict(zip(self.pipeline_names, self.samples_distribution)),
            'is_fitted': self.is_fitted_
        }
        
        if self.is_fitted_:
            summary['actual_samples'] = len(self.sample_results_)
            summary['samples_by_pipeline'] = {}
            for name in self.pipeline_names:
                count = sum(1 for s in self.sample_results_ if s['pipeline_name'] == name)
                summary['samples_by_pipeline'][name] = count
        
        return summary
    
    def __repr__(self):
        """String representation of the MultiPipelineBootstrapper."""
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (f"MultiPipelineBootstrapper(n_pipelines={self.n_pipelines}, "
                f"n_samples={self.n_samples}, status={status})")