Quick Start Guide
=================

This guide will get you started with treeclust in just a few minutes.

Installation
------------

Install treeclust using pip:

.. code-block:: bash

   # Basic installation
   pip install treeclust
   
   # Recommended: install with clustering algorithms
   pip install "treeclust[clustering]"
   
   # For biological data analysis
   pip install "treeclust[clustering,bio]"
   
   # Full installation with all features
   pip install "treeclust[all]"

Basic Usage
-----------

Here's a simple example showing the core workflow:

.. code-block:: python

   import numpy as np
   from treeclust.neighbors import KNeighbors
   from treeclust.clustering import Leiden
   
   # 1. Generate or load your data
   X = np.random.randn(500, 20)  # 500 samples, 20 features
   
   # 2. Create adjacency matrix from your data
   knn = KNeighbors(n_neighbors=15, mode='connectivity')
   adjacency = knn.fit_transform(X)
   
   # 3. Perform clustering
   leiden = Leiden(resolution=1.0)
   labels = leiden.fit_predict(adjacency)
   
   # 4. Analyze results
   n_clusters = len(np.unique(labels))
   print(f"Found {n_clusters} clusters")

Key Concepts
------------

TreeClust follows a modular design with clear separation of concerns:

**1. Neighbors Computation**
   Create adjacency matrices that define the connectivity between data points.

**2. Clustering** 
   Apply clustering algorithms to the adjacency matrices.

**3. Analysis**
   Evaluate clustering quality and stability.

Core Workflow
-------------

Step 1: Neighbors Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from treeclust.neighbors import KNeighbors
   
   # Basic k-nearest neighbors
   knn = KNeighbors(n_neighbors=15, metric='euclidean')
   adjacency = knn.fit_transform(X, mode='connectivity')
   
   # For high-dimensional data, consider shared neighbors
   from treeclust.neighbors import MutualNearestNeighbors
   sn = MutualNearestNeighbors(n_neighbors=20, similarity='jaccard')
   adjacency = sn.fit_transform(X)

Step 2: Clustering
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from treeclust.clustering import Leiden, Louvain
   
   # Leiden clustering (recommended)
   leiden = Leiden(resolution=1.0, n_repetitions=5)
   labels = leiden.fit_predict(adjacency)
   
   # Check clustering consistency
   consistency = leiden.get_consistency_summary()
   print(f"Mean consistency: {consistency['mean_consistency']:.3f}")

Step 3: Parameter Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from treeclust.clustering import MultiresolutionLeiden
   
   # Explore multiple resolution values
   mr_leiden = MultiresolutionLeiden(
       resolution_values=[0.1, 0.5, 1.0, 2.0, 5.0],
       n_repetitions=5
   )
   
   results = mr_leiden.fit_predict(adjacency)
   
   # Find best resolution
   best_resolution = mr_leiden.get_best_resolution('consistency')
   best_labels = mr_leiden.get_labels(best_resolution)

Advanced Features
-----------------

Consistency Checking
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run clustering multiple times to check stability
   leiden = Leiden(
       resolution=1.0,
       n_repetitions=10,           # Run 10 times
       consistency_metric='ari'    # Use Adjusted Rand Index
   )
   
   labels = leiden.fit_predict(adjacency)
   
   # Get detailed consistency analysis
   summary = leiden.get_consistency_summary()
   print(f"Mean ARI: {summary['mean_consistency']:.3f}")
   print(f"Std ARI: {summary['std_consistency']:.3f}")

Robust Neighbors
~~~~~~~~~~~~~~~~

.. code-block:: python

   from treeclust.neighbors import ConsensusNearestNeighbors
   
   # Consensus across multiple parameter settings
   consensus_knn = ConsensusNearestNeighbors(
       n_neighbors_list=[10, 15, 20],    # Try different k values
       n_bootstrap=50,                   # Bootstrap sampling
       consensus_threshold=0.5           # Threshold for consensus
   )
   
   robust_adjacency = consensus_knn.fit_transform(X)

Generic Parameter Tuning
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from treeclust.clustering import Multiscale, Leiden
   
   # Create base clustering instance
   leiden_base = Leiden(resolution=1.0, random_state=42)
   
   # Tune any parameter
   multiscale = Multiscale(
       clustering_instance=leiden_base,
       parameter_name='resolution',
       parameter_values=[0.1, 0.5, 1.0, 2.0, 5.0]
   )
   
   results = multiscale.fit_predict(adjacency)

Common Patterns
---------------

Pattern 1: Basic Clustering Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from treeclust.neighbors import KNeighbors  
   from treeclust.clustering import Leiden
   
   def cluster_data(X, n_neighbors=15, resolution=1.0):
       """Simple clustering pipeline."""
       # Create adjacency
       knn = KNeighbors(n_neighbors=n_neighbors, mode='connectivity')
       adjacency = knn.fit_transform(X)
       
       # Cluster
       leiden = Leiden(resolution=resolution)
       labels = leiden.fit_predict(adjacency)
       
       return labels, adjacency

Pattern 2: Robust Parameter Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from treeclust.clustering import MultiresolutionLeiden
   from treeclust.neighbors import ConsensusNearestNeighbors
   
   def robust_clustering(X, resolution_values=None):
       """Robust clustering with parameter exploration."""
       if resolution_values is None:
           resolution_values = [0.1, 0.5, 1.0, 1.5, 2.0]
       
       # Robust neighbors
       consensus_knn = ConsensusNearestNeighbors(
           n_neighbors_list=[10, 15, 20],
           n_bootstrap=30
       )
       adjacency = consensus_knn.fit_transform(X)
       
       # Multi-resolution clustering
       mr_leiden = MultiresolutionLeiden(
           resolution_values=resolution_values,
           n_repetitions=5
       )
       results = mr_leiden.fit_predict(adjacency)
       
       # Select best resolution
       best_res = mr_leiden.get_best_resolution('consistency')
       return mr_leiden.get_labels(best_res), mr_leiden

Pattern 3: High-Dimensional Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from treeclust.neighbors import MutualNearestNeighbors
   from treeclust.clustering import Leiden
   
   def cluster_high_dim(X, n_neighbors=20, min_shared=3):
       """Clustering for high-dimensional data."""
       # Use shared neighbors for high dimensions
       sn = MutualNearestNeighbors(
           n_neighbors=n_neighbors,
           similarity='jaccard', 
           min_shared=min_shared
       )
       adjacency = sn.fit_transform(X)
       
       # Cluster with consistency checking
       leiden = Leiden(resolution=1.0, n_repetitions=10)
       labels = leiden.fit_predict(adjacency)
       
       return labels, leiden.get_consistency_summary()

Tips and Best Practices
------------------------

**1. Choose Appropriate Parameters**

- **n_neighbors**: Start with 10-20, increase for larger datasets
- **resolution**: Start with 1.0, explore 0.1-5.0 range
- **n_repetitions**: Use 5-10 for consistency checking

**2. Data Preprocessing**

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   
   # Normalize features for better distance computation
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

**3. Handle Large Datasets**

For datasets with > 10,000 samples, consider:

- Using sparse adjacency matrices (mode='connectivity')
- Reducing n_repetitions for speed
- Using approximate neighbors methods

**4. Validate Results**

.. code-block:: python

   # Always check consistency for stochastic methods
   if hasattr(clusterer, 'get_consistency_summary'):
       summary = clusterer.get_consistency_summary()
       if summary['mean_consistency'] < 0.7:
           print("Warning: Low clustering consistency")

Next Steps
----------

- Read the :doc:`../api/index` for detailed API documentation
- Explore :doc:`examples/index` for more complex use cases
- Check :doc:`tutorials/index` for in-depth tutorials
- See the `GitHub repository <https://github.com/gatocor/treeclust>`_ for latest updates

Common Issues
-------------

**ImportError: leidenalg not found**
   Install optional dependencies: ``pip install leidenalg igraph``

**Memory issues with large datasets**
   Use sparse matrices and reduce n_repetitions

**Poor clustering quality**
   Try different n_neighbors values or use MutualNearestNeighbors for high-dimensional data

**Inconsistent results**
   Increase n_repetitions and check consistency metrics