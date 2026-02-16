# Splitter wrappers for Scanpy

import numpy as np
from typing import Optional, Tuple, Union
try:
    import anndata as ad
except ImportError:
    ad = None

from treeclust.validation.data_splitter import PoissonDESplitter, NegativeBinomialDESplitter

def poisson_de_split(
    adata: 'ad.AnnData',
    epsilon: float = 0.8,
    layer: Optional[str] = None,
    use_raw: bool = False,
    random_state: Optional[int] = None,
    key_added_train: str = "poisson_de_split_train",
    key_added_test: str = "poisson_de_split_test",
    copy: bool = False
) -> Optional['ad.AnnData']:
    """
    Split count data using Poisson DE bootstrapping for differential expression analysis.
    
    This function applies binomial sampling to each entry in the count matrix,
    creating bootstrap and test samples that preserve the count structure while
    enabling robust differential expression analysis. The splits are stored as
    new layers in the AnnData object, preserving the original data.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing count matrix.
    epsilon : float, default=0.8
        Probability parameter for binomial sampling (0 < epsilon <= 1).
        Higher values retain more counts in the bootstrap sample.
    layer : str, optional
        Layer of adata to use. If None, uses adata.X.
    use_raw : bool, default=False
        Whether to use adata.raw.X instead of adata.X.
    random_state : int, optional
        Random seed for reproducibility.
    key_added_train : str, default="poisson_de_split_train"
        Key for the bootstrap/training split layer.
    key_added_test : str, default="poisson_de_split_test"
        Key for the test split layer.
    copy : bool, default=False
        Whether to return a copy of adata or modify in place.
        
    Returns:
    --------
    adata : AnnData or None
        If copy=True, returns AnnData object with bootstrap and test splits stored as layers.
        If copy=False, modifies adata in place and returns None.
        
    Examples:
    ---------
    >>> import scanpy as sc
    >>> from treeclust.scanpy_wrappers import poisson_de_split
    >>> 
    >>> # Load data
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> 
    >>> # Option 1: In-place modification (default)
    >>> poisson_de_split(adata, epsilon=0.7)  # Returns None, modifies adata
    >>> bootstrap_data = adata.layers['poisson_de_split_train']
    >>> 
    >>> # Option 2: Return copy
    >>> adata_with_splits = poisson_de_split(adata, epsilon=0.7, copy=True)
    >>> test_data = adata_with_splits.layers['poisson_de_split_test']
    """
    if ad is None:
        raise ImportError("anndata is required for Scanpy wrappers. Install with: pip install anndata")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Copy the data if requested
    if copy:
        adata = adata.copy()
    
    # Get the count matrix
    if use_raw:
        if adata.raw is None:
            raise ValueError("adata.raw is None, cannot use use_raw=True")
        X = adata.raw.X
    elif layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]
    else:
        X = adata.X
    
    # Create bootstrapper and sample
    bootstrapper = PoissonDESplitter(epsilon=epsilon)
    X_bootstrap, X_test = bootstrapper.sample(X)
    
    # Store the splits as new layers (preserving original data)
    adata.layers[key_added_train] = X_bootstrap
    adata.layers[key_added_test] = X_test
    
    # Add metadata about the split
    adata.uns['poisson_de_split'] = {
        'epsilon': epsilon,
        'layer_used': layer if layer is not None else ('raw' if use_raw else 'X'),
        'key_train': key_added_train,
        'key_test': key_added_test,
        'random_state': random_state
    }

    if copy:
        return adata
    else:
        return None

def negativebinomial_de_split(
    adata: 'ad.AnnData',
    epsilon: float = 0.8,
    layer: Optional[str] = None,
    use_raw: bool = False,
    random_state: Optional[int] = None,
    key_added_train: str = "negativebinomial_de_split_train",
    key_added_test: str = "negativebinomial_de_split_test",
    copy: bool = False
) -> Optional['ad.AnnData']:
    """
    Split count data using Negative Binomial DE bootstrapping for differential expression analysis.
    
    This function applies Dirichlet-Multinomial sampling to each entry in the count matrix,
    creating bootstrap and test samples with overdispersion that's more suitable for
    negative binomial-distributed count data. The splits are stored as new layers in 
    the AnnData object, preserving the original data.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing count matrix.
    epsilon : float, default=0.8
        Weight parameter for Dirichlet-Multinomial sampling (0 < epsilon <= 1).
        Higher values favor the bootstrap sample. Concentration parameters are [epsilon, 1-epsilon].
    layer : str, optional
        Layer of adata to use. If None, uses adata.X.
    use_raw : bool, default=False
        Whether to use adata.raw.X instead of adata.X.
    random_state : int, optional
        Random seed for reproducibility.
    key_added_train : str, default="negativebinomial_de_split_train"
        Key for the bootstrap/training split layer.
    key_added_test : str, default="negativebinomial_de_split_test"
        Key for the test split layer.
    copy : bool, default=False
        Whether to return a copy of adata or modify in place.
        
    Returns:
    --------
    adata : AnnData or None
        If copy=True, returns AnnData object with bootstrap and test splits stored as layers.
        If copy=False, modifies adata in place and returns None.
        
    Examples:
    ---------
    >>> import scanpy as sc
    >>> from treeclust.scanpy_wrappers import negativebinomial_de_split
    >>> 
    >>> # Load data
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> 
    >>> # Option 1: In-place modification (default)
    >>> negativebinomial_de_split(adata, epsilon=0.7)  # Returns None, modifies adata
    >>> bootstrap_data = adata.layers['negativebinomial_de_split_train']
    >>> 
    >>> # Option 2: Return copy
    >>> adata_with_splits = negativebinomial_de_split(adata, epsilon=0.7, copy=True)
    >>> test_data = adata_with_splits.layers['negativebinomial_de_split_test']
    """
    if ad is None:
        raise ImportError("anndata is required for Scanpy wrappers. Install with: pip install anndata")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Copy the data if requested
    if copy:
        adata = adata.copy()
    
    # Get the count matrix
    if use_raw:
        if adata.raw is None:
            raise ValueError("adata.raw is None, cannot use use_raw=True")
        X = adata.raw.X
    elif layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]
    else:
        X = adata.X
    
    # Create bootstrapper and sample
    bootstrapper = NegativeBinomialDESplitter(epsilon=epsilon)
    X_bootstrap, X_test = bootstrapper.sample(X)
    
    # Store the splits as new layers (preserving original data)
    adata.layers[key_added_train] = X_bootstrap
    adata.layers[key_added_test] = X_test
    
    # Add metadata about the split
    adata.uns['nb_de_split'] = {
        'epsilon': epsilon,
        'layer_used': layer if layer is not None else ('raw' if use_raw else 'X'),
        'key_train': key_added_train,
        'key_test': key_added_test,
        'random_state': random_state
    }
    
    if copy:
        return adata
    else:
        return None