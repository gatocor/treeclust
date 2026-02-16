"""
Scanpy-style preprocessing functions for treeclust.

This module provides scanpy-compatible preprocessing functions following
the same patterns as scanpy.pp for seamless integration.
"""

from .neighbors import (
    neighbors,
    consensus_neighbors,
    mutual_neighbors,
    consensus_mutual_neighbors,
    coassociation_matrix
)

__all__ = [
    'neighbors',
    'consensus_neighbors', 
    'mutual_neighbors',
    'consensus_mutual_neighbors',
    'coassociation_matrix'
]