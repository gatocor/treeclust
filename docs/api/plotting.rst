Plotting (:mod:`treeclust.plotting`)
===================================

.. automodule:: treeclust.plotting

The plotting module provides visualization tools for clustering results, adjacency matrices,
and analysis of clustering quality across parameter ranges.

(Module currently under development - plotting functions will be added in future releases)

Planned Features
----------------

**Cluster Visualization**
   - 2D/3D scatter plots of clustered data
   - Cluster boundary visualization
   - Hierarchical cluster dendrograms

**Adjacency Matrix Visualization**  
   - Heatmaps of adjacency matrices
   - Network graph visualization
   - Connectivity pattern analysis

**Parameter Analysis Plots**
   - Resolution vs number of clusters
   - Consistency scores across parameters  
   - Quality metrics comparison plots

**Statistical Plots**
   - Cluster size distributions
   - Silhouette analysis plots
   - Bootstrap stability visualization

Example Usage (Future)
----------------------

.. code-block:: python

   from treeclust.plotting import plot_clusters, plot_resolution_analysis
   from treeclust.clustering import MultiresolutionLeiden
   import numpy as np
   
   # Multi-resolution clustering
   mr_leiden = MultiresolutionLeiden(
       resolution_values=[0.1, 0.5, 1.0, 2.0, 5.0]
   )
   results = mr_leiden.fit_predict(adjacency)
   
   # Plot clustering results  
   plot_clusters(X, labels, title='Leiden Clustering Results')
   
   # Plot resolution analysis
   plot_resolution_analysis(mr_leiden, metrics=['n_clusters', 'modularity'])

Contributing
------------

The plotting module is actively being developed. Contributions of visualization functions
are welcome. Please see the contributing guidelines for more information.