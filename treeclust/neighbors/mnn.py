"""
Shared Neighbors class for treeclust.

This module provides the MutualNearestNeighbors class that computes shared neighbor
connectivity based on Jaccard index.
"""

import numpy as np
from scipy import sparse
import scipy.sparse as sp
from .sklearn import NearestNeighbors

class MutualNearestNeighbors:
    """
    Mutual Nearest Neighbors class using mutual nearest neighbors.
    
    This class computes shared neighbor connectivity by finding mutual nearest neighbors:
    nodes that are nearest neighbors of each other (reciprocal connections).
    
    Features:
    - Computes k-nearest neighbors connectivity
    - Keeps only reciprocal/mutual connections where both (i,j) and (j,i) exist
    - Returns binary connectivity matrix for mutual nearest neighbors
    """
    
    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = 'euclidean',
        flavor: str = 'auto'
    ):
        """
        Initialize the MutualNearestNeighbors class.
        
        Parameters:
        -----------
        n_neighbors : int, default=15
            Number of neighbors to use for initial neighborhood computation.
            
        metric : str, default='euclidean'
            Distance metric to use for finding initial neighbors.
            
        flavor : str, default='auto'
            Backend implementation to use ('auto', 'cuml', 'sklearn').
        """
        # Store all parameters as instance attributes
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.flavor = flavor
        
        # Initialize state
        self.is_fitted_ = False
                        
    def fit(self, X: np.ndarray) -> 'MutualNearestNeighbors':
        """
        Fit the shared neighbors model.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).
            
        Returns:
        --------
        self : MutualNearestNeighbors
            Returns self for method chaining.
        """
        # First, compute initial k-nearest neighbors connectivity matrix
        knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )
        knn.fit(X)
        connectivity_matrix = knn.kneighbors_graph(X, mode='connectivity')
        
        # Find reciprocal connections (mutual nearest neighbors)
        # An edge (i,j) is kept only if both (i,j) and (j,i) are present in the KNN matrix
        reciprocal_matrix = connectivity_matrix.multiply(connectivity_matrix.T)
        
        # Store as sparse matrix - reciprocal_matrix already contains only mutual connections
        self.matrix_ = reciprocal_matrix.astype(float)
        self.is_fitted_ = True
        
        return self
    
    def fit_transform(self, X):
        """Fit the model and return the shared neighbors connectivity matrix."""
        self.fit(X)
        return self.matrix_
        
    def __repr__(self) -> str:
        """String representation of the MutualNearestNeighbors object."""        
        return (f"MutualNearestNeighbors(n_neighbors={self.n_neighbors}, "
                f"metric='{self.metric}', "
                f"flavor='{self.flavor}', "
                f"fitted={self.is_fitted_})")