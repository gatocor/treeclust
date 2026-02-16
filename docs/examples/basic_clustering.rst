Basic Clustering
================

This example demonstrates the fundamental clustering workflow in treeclust.

Simple Clustering Example
--------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from treeclust.neighbors import KNeighbors
   from treeclust.clustering import Leiden
   
   # Generate sample data with clear clusters
   np.random.seed(42)
   
   # Create three blob clusters
   cluster1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
   cluster2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 100) 
   cluster3 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], 100)
   
   X = np.vstack([cluster1, cluster2, cluster3])
   
   # Step 1: Create adjacency matrix
   knn = KNeighbors(n_neighbors=10, mode='connectivity')
   adjacency = knn.fit_transform(X)
   
   print(f"Adjacency matrix shape: {adjacency.shape}")
   print(f"Number of edges: {adjacency.nnz}")
   
   # Step 2: Perform clustering
   leiden = Leiden(resolution=1.0, random_state=42)
   labels = leiden.fit_predict(adjacency)
   
   # Step 3: Analyze results
   n_clusters = len(np.unique(labels))
   print(f"Number of clusters found: {n_clusters}")
   print(f"Cluster sizes: {np.bincount(labels)}")
   
   # Visualize results (if 2D data)
   plt.figure(figsize=(10, 4))
   
   plt.subplot(1, 2, 1)
   plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
   plt.title('Original Data')
   plt.xlabel('Feature 1')
   plt.ylabel('Feature 2')
   
   plt.subplot(1, 2, 2)
   plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
   plt.title('Leiden Clustering Results')
   plt.xlabel('Feature 1') 
   plt.ylabel('Feature 2')
   plt.colorbar()
   
   plt.tight_layout()
   plt.show()

Output::

   Adjacency matrix shape: (300, 300)
   Number of edges: 3000
   Number of clusters found: 3
   Cluster sizes: [100 100 100]

Comparing Different Algorithms
------------------------------

.. code-block:: python

   from treeclust.clustering import Leiden, Louvain
   import numpy as np
   
   # Same data as above
   
   # Compare Leiden vs Louvain
   algorithms = {
       'Leiden': Leiden(resolution=1.0, random_state=42),
       'Louvain': Louvain(resolution=1.0, random_state=42)
   }
   
   results = {}
   for name, algorithm in algorithms.items():
       labels = algorithm.fit_predict(adjacency)
       n_clusters = len(np.unique(labels))
       results[name] = {
           'labels': labels,
           'n_clusters': n_clusters,
           'cluster_sizes': np.bincount(labels)
       }
   
   # Compare results
   for name, result in results.items():
       print(f"{name}:")
       print(f"  Clusters: {result['n_clusters']}")
       print(f"  Sizes: {result['cluster_sizes']}")

Real Data Example
-----------------

.. code-block:: python

   from sklearn.datasets import load_iris
   from treeclust.neighbors import KNeighbors
   from treeclust.clustering import Leiden
   import numpy as np
   
   # Load iris dataset
   iris = load_iris()
   X = iris.data
   true_labels = iris.target
   
   print(f"Data shape: {X.shape}")
   print(f"True number of classes: {len(np.unique(true_labels))}")
   
   # Cluster the data
   knn = KNeighbors(n_neighbors=5, mode='connectivity')
   adjacency = knn.fit_transform(X)
   
   leiden = Leiden(resolution=0.8, random_state=42)
   pred_labels = leiden.fit_predict(adjacency)
   
   print(f"Predicted clusters: {len(np.unique(pred_labels))}")
   
   # Compare with true labels using ARI
   from sklearn.metrics import adjusted_rand_score
   ari = adjusted_rand_score(true_labels, pred_labels)
   print(f"Adjusted Rand Index: {ari:.3f}")

Working with Different Data Types
----------------------------------

High-Dimensional Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from treeclust.neighbors import MutualNearestNeighbors
   from treeclust.clustering import Leiden
   
   # Generate high-dimensional data
   np.random.seed(42)
   n_samples, n_features = 200, 100
   X = np.random.randn(n_samples, n_features)
   
   # Add some structure
   X[:50, :20] += 3    # First cluster
   X[50:100, 20:40] += 3  # Second cluster
   
   # Use shared neighbors for high-dimensional data
   sn = MutualNearestNeighbors(
       n_neighbors=15,
       similarity='jaccard',
       min_shared=3
   )
   adjacency = sn.fit_transform(X)
   
   # Cluster
   leiden = Leiden(resolution=1.0)
   labels = leiden.fit_predict(adjacency)
   
   print(f"Clusters in high-dim data: {len(np.unique(labels))}")

Sparse Data
~~~~~~~~~~~

.. code-block:: python

   from scipy.sparse import random
   from treeclust.neighbors import KNeighbors
   from treeclust.clustering import Leiden
   
   # Generate sparse data matrix
   X_sparse = random(500, 1000, density=0.1, format='csr')
   
   # treeclust handles sparse data naturally
   knn = KNeighbors(n_neighbors=10, mode='connectivity')
   adjacency = knn.fit_transform(X_sparse)
   
   leiden = Leiden(resolution=1.5)
   labels = leiden.fit_predict(adjacency)
   
   print(f"Clusters in sparse data: {len(np.unique(labels))}")

Key Takeaways
-------------

1. **Modular Design**: Separate neighbors computation from clustering
2. **Flexible Input**: Works with dense, sparse, and high-dimensional data  
3. **Parameter Impact**: Resolution parameter controls cluster granularity
4. **Algorithm Choice**: Leiden generally preferred over Louvain for quality
5. **Data Preprocessing**: Consider normalization for better distance computation

Next Steps
----------

- Try :doc:`multiresolution_analysis` for parameter exploration
- See :doc:`consensus_methods` for robust clustering
- Read :doc:`../tutorials/clustering_concepts` for deeper understanding