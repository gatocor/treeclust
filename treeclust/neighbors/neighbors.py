"""
K-Nearest Neighbors utilities for treeclust.

This module provides imports for all neighbor classes from their modular files.
"""

# Import all neighbor classes from their separate files
from .kneighbors import KNeighbors
from .cnn import ConsensusKNeighbors  
from .mnn import SharedNeighbors
from .cmnn import ConsensusMutualNearestNeighbors

# Import sklearn-style wrapper functions
from .sklearn import NearestNeighbors, kneighbors_graph

__all__ = [
    'KNeighbors',
    'ConsensusKNeighbors', 
    'SharedNeighbors',
    'ConsensusMutualNearestNeighbors',
    'NearestNeighbors',
    'kneighbors_graph'
]

