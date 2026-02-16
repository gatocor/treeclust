"""
Metrics module for treeclust.

This module provides clustering evaluation metrics and consistency measures.
"""

from .metrics import connectivity_probability

__all__ = [
    'connectivity_probability'
]