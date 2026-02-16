"""
Scanpy-style tools for treeclust.

This module provides scanpy-compatible clustering and analysis functions 
following the same patterns as scanpy.tl for seamless integration.
"""

from .clustering import (
    leiden,
    louvain,
    multiresolution_leiden,
    multiresolution_louvain
)

from .annotation import (
    annotate,
    propagate_annotation,
    annotate_hierarchical
)

__all__ = [
    'leiden',
    'louvain', 
    'multiresolution_leiden',
    'multiresolution_louvain',
    'annotate',
    'propagate_annotation',
    'annotate_hierarchical'
]