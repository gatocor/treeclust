"""
Scanpy-compatible interface for treeclust.

This module provides scanpy-style functions and interfaces to make treeclust
methods easily accessible to users familiar with scanpy's API.
"""

from . import pp
from . import tl
from . import pl

__all__ = ['pp', 'tl', 'pl']