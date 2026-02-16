TreeClust Documentation
========================

TreeClust is a Python package for robust clustering in a hierarchical manner, providing various clustering algorithms with sklearn-style interfaces and support for GPU acceleration.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples/index
   tutorials/index

Features
--------

- **Multiple clustering algorithms**: Leiden, Louvain with consistency checking
- **Multiresolution analysis**: Specialized classes for resolution parameter tuning
- **Modular design**: Separate neighbors computation from clustering algorithms
- **Consistency validation**: Statistical validation for stochastic clustering methods
- **sklearn-style interface**: Familiar fit/predict pattern for easy integration
- **GPU acceleration**: Optional GPU support for large-scale clustering
- **Comprehensive metrics**: Built-in clustering quality metrics and analysis tools

Quick Example
-------------

.. code-block:: python

   from treeclust.clustering import Leiden
   from treeclust.neighbors import KNeighbors
   import numpy as np

   # Generate sample data
   X = np.random.randn(1000, 50)

   # Create adjacency matrix
   knn = KNeighbors(n_neighbors=15, mode='connectivity')
   adjacency = knn.fit_transform(X)

   # Perform Leiden clustering
   leiden = Leiden(resolution=1.0, n_repetitions=5)
   labels = leiden.fit_predict(adjacency)

   # Get consistency summary
   consistency = leiden.get_consistency_summary()
   print(f"Mean consistency: {consistency['mean_consistency']:.3f}")

Installation
------------

Install treeclust from PyPI:

.. code-block:: bash

   # Basic installation
   pip install treeclust
   
   # Recommended: with clustering algorithms  
   pip install "treeclust[clustering]"

.. code-block:: bash

   # Full installation with all optional features
   pip install "treeclust[all]"

For more installation options, see the :doc:`installation` guide.

Modules
-------

treeclust.clustering
   Core clustering algorithms including Leiden, Louvain, and multiresolution variants.

treeclust.neighbors 
   Neighbors computation and adjacency matrix construction.

treeclust.metrics
   Clustering quality metrics and evaluation tools.

treeclust.decomposition
   Dimensionality reduction and manifold learning algorithms.

treeclust.plotting
   Visualization tools for clustering results and analysis.

treeclust.utils
   Utility functions and helper tools.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`