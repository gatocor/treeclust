"""
Example demonstrating the Louvain clustering class.

This example shows how to use the new Louvain class for community detection
with both CPU and GPU implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Import TreeClust classes
from treeclust import Louvain

# Generate synthetic data for testing
np.random.seed(42)
X, true_labels = make_blobs(n_samples=200, centers=4, n_features=2, 
                           cluster_std=1.0, random_state=42)

print("Generated synthetic data:")
print(f"  Data shape: {X.shape}")
print(f"  True number of clusters: {len(np.unique(true_labels))}")

# ===================================================================
# Example 1: Basic Louvain Clustering with Auto Flavor
# ===================================================================

print("\n" + "="*60)
print("Example 1: Basic Louvain Clustering (Auto Flavor)")
print("="*60)

# Create Louvain clusterer with auto flavor (uses GPU if available)
louvain_auto = Louvain(
    resolution=1.0,
    random_state=42,
    flavor='auto'
)

print(f"Louvain instance: {louvain_auto}")

# Fit and predict
labels_auto = louvain_auto.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels_auto, cmap='tab10', alpha=0.7, s=50)
plt.show()