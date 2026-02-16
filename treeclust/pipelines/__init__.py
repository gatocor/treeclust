"""
Pipelines module for treeclust.

This module provides pipeline classes and functions that combine bootstrapping
with various machine learning and data processing steps.
"""

# Import pipeline functions (sklearn is required dependency)
from .pipelines import (
    PipelineBootstrapper,
    pipeline_matrix_bootstrap_pca,
    pipeline_matrix_bootstrap_vae,
    MultiPipelineBootstrapper,
    BootstrapTransformIterator,
    BootstrapPredictIterator
)

__all__ = [
    'PipelineBootstrapper',
    'pipeline_matrix_bootstrap_pca',
    'pipeline_matrix_bootstrap_vae',
    'MultiPipelineBootstrapper',
    'BootstrapTransformIterator',
    'BootstrapPredictIterator'
]