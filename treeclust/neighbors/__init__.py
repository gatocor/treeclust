"""
Neighbors module for treeclust.

This module provides K-nearest neighbors and shared neighbors algorithms
with sklearn/cuML compatibility and consensus/bootstrapping capabilities.
"""

from .sklearn import NearestNeighbors
from .cnn import ConsensusNearestNeighbors  
from .mnn import MutualNearestNeighbors
from .cmnn import ConsensusMutualNearestNeighbors
from .coassotiation import CoassociationDistanceMatrix
from .utils import to_connectivities_matrix, to_transition_matrix

# Import sklearn-style wrapper functions
from .sklearn import NearestNeighbors

__all__ = [
    'ConsensusNearestNeighbors', 
    'MutualNearestNeighbors',
    'ConsensusMutualNearestNeighbors',
    'CoassociationDistanceMatrix',
    'NearestNeighbors',
    'to_connectivities_matrix',
    'to_transition_matrix'
]
