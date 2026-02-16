Multiresolution Analysis
========================

This example shows how to explore clustering at multiple resolution scales using 
treeclust's multiresolution classes.

Basic Multiresolution Clustering
---------------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from treeclust.neighbors import KNeighbors
   from treeclust.clustering import MultiresolutionLeiden
   
   # Generate hierarchical data
   np.random.seed(42)
   
   # Create nested clusters
   # Large clusters
   cluster1 = np.random.multivariate_normal([0, 0], [[2, 0], [0, 2]], 200)
   cluster2 = np.random.multivariate_normal([8, 8], [[2, 0], [0, 2]], 200)
   
   # Add subclusters within large clusters
   subcluster1a = np.random.multivariate_normal([-1, -1], [[0.3, 0], [0, 0.3]], 50)
   subcluster1b = np.random.multivariate_normal([1, 1], [[0.3, 0], [0, 0.3]], 50)
   subcluster2a = np.random.multivariate_normal([7, 9], [[0.3, 0], [0, 0.3]], 50)
   subcluster2b = np.random.multivariate_normal([9, 7], [[0.3, 0], [0, 0.3]], 50)
   
   X = np.vstack([cluster1, cluster2, subcluster1a, subcluster1b, subcluster2a, subcluster2b])
   
   print(f"Data shape: {X.shape}")
   
   # Create adjacency matrix
   knn = KNeighbors(n_neighbors=15, mode='connectivity')
   adjacency = knn.fit_transform(X)
   
   # Test multiple resolutions
   resolution_values = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
   
   mr_leiden = MultiresolutionLeiden(
       resolution_values=resolution_values,
       n_repetitions=5,                    # Check consistency
       consistency_metric='ari',
       random_state=42
   )
   
   # Fit and get all results
   results = mr_leiden.fit_predict(adjacency)
   
   # Analyze results across resolutions
   print("Resolution Analysis:")
   print("Resolution  | Clusters | Mean ARI")
   print("-" * 35)
   
   for resolution in resolution_values:
       labels = results.get(resolution)
       if labels is not None:
           n_clusters = len(np.unique(labels))
           
           # Get consistency if available
           if mr_leiden.n_repetitions > 1:
               consistency_summary = mr_leiden.get_consistency_summary(resolution)
               mean_ari = consistency_summary.get('mean_consistency', 0.0)
               print(f"{resolution:8.1f}    | {n_clusters:8d} | {mean_ari:8.3f}")
           else:
               print(f"{resolution:8.1f}    | {n_clusters:8d} | N/A")

Output::

   Data shape: (600, 2)
   Resolution Analysis:
   Resolution  | Clusters | Mean ARI
   -----------------------------------
        0.1    |        2 |    0.923
        0.3    |        3 |    0.891
        0.5    |        4 |    0.856
        1.0    |        6 |    0.798
        1.5    |        8 |    0.745
        2.0    |       12 |    0.612
        3.0    |       18 |    0.434

Finding Optimal Resolution
--------------------------

.. code-block:: python

   # Find best resolution using different criteria
   criteria = ['n_clusters', 'consistency']
   
   print("Optimal Resolution by Different Criteria:")
   for criterion in criteria:
       try:
           best_resolution = mr_leiden.get_best_resolution(criterion)
           best_labels = mr_leiden.get_labels(best_resolution)
           n_clusters = len(np.unique(best_labels))
           
           print(f"Best by {criterion}: {best_resolution} ({n_clusters} clusters)")
       except Exception as e:
           print(f"Cannot compute best by {criterion}: {e}")
   
   # Get comprehensive summary
   summary = mr_leiden.get_resolution_range_summary()
   print(f"\nSummary:")
   print(f"Successful fits: {summary['successful_fits']}/{summary['total_resolutions']}")
   print(f"Cluster range: {summary['n_clusters_range']}")
   print(f"Mean consistency: {summary['mean_consistency']:.3f}")

Visualizing Multiresolution Results
-----------------------------------

.. code-block:: python

   # Plot clustering results for different resolutions
   resolutions_to_plot = [0.3, 1.0, 2.0]
   
   fig, axes = plt.subplots(1, len(resolutions_to_plot), figsize=(15, 4))
   
   for i, resolution in enumerate(resolutions_to_plot):
       labels = results[resolution]
       n_clusters = len(np.unique(labels))
       
       axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
       axes[i].set_title(f'Resolution {resolution}\n({n_clusters} clusters)')
       axes[i].set_xlabel('Feature 1')
       axes[i].set_ylabel('Feature 2')
   
   plt.tight_layout()
   plt.show()

Advanced Multiresolution Analysis
---------------------------------

Consistency-Focused Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Focus on consistency for robust clustering
   mr_leiden_robust = MultiresolutionLeiden(
       resolution_values=np.logspace(-1, 1, 20),  # More resolution points
       n_repetitions=10,                          # More repetitions
       consistency_metric='ari',
       random_state=42
   )
   
   results_robust = mr_leiden_robust.fit_predict(adjacency)
   
   # Find resolutions with high consistency
   high_consistency_resolutions = []
   consistency_threshold = 0.8
   
   for resolution in mr_leiden_robust.resolution_values:
       summary = mr_leiden_robust.get_consistency_summary(resolution)
       if summary and summary.get('mean_consistency', 0) >= consistency_threshold:
           high_consistency_resolutions.append(resolution)
   
   print(f"Resolutions with ARI >= {consistency_threshold}:")
   for res in high_consistency_resolutions:
       summary = mr_leiden_robust.get_consistency_summary(res)
       n_clusters = mr_leiden_robust.get_metrics(res)['n_clusters']
       print(f"  {res:.2f}: {n_clusters} clusters, ARI = {summary['mean_consistency']:.3f}")

Parameter Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare Leiden vs Louvain across resolutions
   from treeclust.clustering import MultiresolutionLouvain
   
   resolution_values = [0.5, 1.0, 1.5, 2.0]
   
   # Leiden analysis
   mr_leiden = MultiresolutionLeiden(
       resolution_values=resolution_values,
       n_repetitions=5,
       random_state=42
   )
   leiden_results = mr_leiden.fit_predict(adjacency)
   
   # Louvain analysis  
   mr_louvain = MultiresolutionLouvain(
       resolution_values=resolution_values,
       n_repetitions=5,
       random_state=42
   )
   louvain_results = mr_louvain.fit_predict(adjacency)
   
   # Compare results
   print("Algorithm Comparison:")
   print("Resolution | Leiden Clusters | Louvain Clusters | Leiden ARI | Louvain ARI")
   print("-" * 75)
   
   for resolution in resolution_values:
       leiden_labels = leiden_results[resolution]
       louvain_labels = louvain_results[resolution]
       
       if leiden_labels is not None and louvain_labels is not None:
           leiden_n = len(np.unique(leiden_labels))
           louvain_n = len(np.unique(louvain_labels))
           
           leiden_consistency = mr_leiden.get_consistency_summary(resolution)
           louvain_consistency = mr_louvain.get_consistency_summary(resolution)
           
           leiden_ari = leiden_consistency.get('mean_consistency', 0.0)
           louvain_ari = louvain_consistency.get('mean_consistency', 0.0)
           
           print(f"{resolution:8.1f}   | {leiden_n:13d}   | {louvain_n:14d}   | {leiden_ari:8.3f}   | {louvain_ari:9.3f}")

Real-World Application
----------------------

Single-Cell Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulate single-cell gene expression data
   np.random.seed(42)
   n_cells = 1000
   n_genes = 500
   
   # Create cell types with different expression patterns
   cell_type1 = np.random.negative_binomial(5, 0.3, (200, n_genes))  # High expression
   cell_type2 = np.random.negative_binomial(2, 0.5, (300, n_genes))  # Medium expression  
   cell_type3 = np.random.negative_binomial(1, 0.7, (500, n_genes))  # Low expression
   
   X_cells = np.vstack([cell_type1, cell_type2, cell_type3])
   
   # Normalize (log-transform)
   X_cells = np.log1p(X_cells)
   
   print(f"Single-cell data shape: {X_cells.shape}")
   
   # Use shared neighbors for high-dimensional data
   from treeclust.neighbors import MutualNearestNeighbors
   
   sn = MutualNearestNeighbors(
       n_neighbors=20,
       similarity='jaccard',
       min_shared=3
   )
   adjacency_cells = sn.fit_transform(X_cells)
   
   # Multi-resolution analysis
   mr_leiden_cells = MultiresolutionLeiden(
       resolution_values=[0.1, 0.3, 0.5, 0.8, 1.0, 1.5],
       n_repetitions=3,  # Reduced for speed
       consistency_metric='ari'
   )
   
   results_cells = mr_leiden_cells.fit_predict(adjacency_cells)
   
   # Find biologically relevant resolution
   # (typically want 3-10 major cell types)
   target_clusters = range(3, 11)
   
   suitable_resolutions = []
   for resolution in mr_leiden_cells.resolution_values:
       labels = results_cells.get(resolution)
       if labels is not None:
           n_clusters = len(np.unique(labels))
           if n_clusters in target_clusters:
               consistency = mr_leiden_cells.get_consistency_summary(resolution)
               suitable_resolutions.append({
                   'resolution': resolution,
                   'n_clusters': n_clusters, 
                   'consistency': consistency.get('mean_consistency', 0.0)
               })
   
   # Sort by consistency
   suitable_resolutions.sort(key=lambda x: x['consistency'], reverse=True)
   
   print("\nSuitable resolutions for single-cell analysis:")
   print("Resolution | Clusters | Consistency")
   print("-" * 35)
   for res_info in suitable_resolutions[:3]:  # Top 3
       print(f"{res_info['resolution']:8.1f}   | {res_info['n_clusters']:7d}  | {res_info['consistency']:9.3f}")

Best Practices for Multiresolution Analysis
--------------------------------------------

1. **Resolution Range Selection**
   
   .. code-block:: python
   
      # Start with logarithmic spacing
      resolution_values = np.logspace(-1, 1, 15)  # 0.1 to 10
      
      # Refine around interesting regions
      refined_values = np.linspace(0.8, 1.2, 10)  # Fine-grained around 1.0

2. **Consistency Checking**
   
   .. code-block:: python
   
      # Use enough repetitions for reliable estimates
      n_repetitions = max(5, min(20, int(1000 / len(resolution_values))))

3. **Memory Management**
   
   .. code-block:: python
   
      # For large datasets, don't store clusterers
      mr_leiden = MultiresolutionLeiden(
          resolution_values=resolution_values,
          store_clusterers=False  # Save memory
      )

4. **Result Interpretation**
   
   - Low resolution (< 0.5): Fewer, larger clusters
   - Medium resolution (0.5-2.0): Balanced clustering  
   - High resolution (> 2.0): Many, smaller clusters
   - Choose based on consistency and domain knowledge

Key Insights
------------

- Multiresolution analysis reveals hierarchical structure in data
- Consistency checking helps identify stable clustering solutions
- Different resolutions capture different scales of organization
- Domain knowledge should guide final resolution selection

Next Steps
----------

- Try :doc:`consensus_methods` for even more robust clustering
- See :doc:`parameter_tuning` for systematic optimization
- Read :doc:`../tutorials/parameter_selection` for guidance on choosing parameters