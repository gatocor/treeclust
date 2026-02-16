"""
Utility functions for treeclust.

This module provides various utility functions including label propagation algorithms
and hierarchical annotation tools.
"""

from .utils import (
    propagate_annotation,
    propagate_labels_iterative,
    annotation_hierarchical,
    annotation_to_hierarchical_clustering_projection
)

__all__ = [
    'propagate_annotation',
    'propagate_labels_iterative',
    'annotation_hierarchical',
    'annotation_to_hierarchical_clustering_projection'
]