"""
Example demonstrating the Leiden clustering class.

This example shows how to use the new Leiden class for community detection
with both CPU and GPU implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Import TreeClust classes
from treeclust import Leiden
from treeclust import ConsensusNearestNeighbors

# Generate synthetic data for testing
np.random.seed(42)
X, true_labels = make_blobs(n_samples=200, centers=4, n_features=2, 
                           cluster_std=1.0, random_state=42)

print("Generated synthetic data:")
print(f"  Data shape: {X.shape}")
print(f"  True number of clusters: {len(np.unique(true_labels))}")

# ===================================================================
# Example 1: Basic Leiden Clustering with Auto Flavor
# ===================================================================

print("\n" + "="*60)
print("Example 1: Basic Leiden Clustering (Auto Flavor)")
print("="*60)

# Create Leiden clusterer with auto flavor (uses GPU if available)

# Fit and predict
for i in range(3):
    leiden = Leiden(
        resolution=0.4,
        random_state=i,
        partition_type='CPM',
        # connectivity=ConsensusNearestNeighbors().fit_transform,
        flavor='auto'
    )

    print(f"Leiden instance: {leiden}")

    labels_auto = leiden.fit_predict(X)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels_auto, cmap='tab10', alpha=0.7, s=50)

plt.show()