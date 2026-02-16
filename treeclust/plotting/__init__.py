"""
Plotting module for treeclust.

This module provides visualization tools for graphs, networks, and clustering results.
"""

from .graph import plot_graph, plot_clustering_comparison, plot_multiresolution_analysis
from .multiresolution_graph import plot_multiresolution_hierarchy
from .multiresolution_clusters import plot_multiresolution_graph
from .multiresolution_dotplot import plot_cluster_expression_dotplot

__all__ = [
    'plot_graph',
    'plot_clustering_comparison', 
    'plot_multiresolution_analysis',
    'plot_multiresolution_hierarchy',
    'plot_multiresolution_graph',
    'plot_cluster_expression_dotplot'
]