"""
Example demonstrating the ClusteringClassBootstrapper class for sampling parameter combinations.

This example shows how to use the ClusteringClassBootstrapper to systematically explore
parameter spaces for robust clustering analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# Import TreeClust classes
from treeclust import Leiden, ConsensusCluster, DataBootstrapper, ClusteringClassBootstrapper

# Generate synthetic data for testing
np.random.seed(42)
X, true_labels = make_blobs(n_samples=200, centers=4, n_features=2, 
                           cluster_std=1.0, random_state=42)

print("Generated synthetic data:")
print(f"  Data shape: {X.shape}")
print(f"  True number of clusters: {len(np.unique(true_labels))}")

# ===================================================================
# Example 1: Basic Parameter DataBootstrapper for Leiden Clustering
# ===================================================================

print("\n" + "="*60)
print("Example 1: Basic Parameter DataBootstrapper for Leiden")
print("="*60)

# Create parameter bootstrapper for Leiden
leiden_bootstrapper = ClusteringClassBootstrapper(
    ml_class=Leiden,
    parameter_ranges= {
            'resolution': (0.1, 2.0),  # Continuous range for resolution parameter
            'random_state': list(range(0, 5)),  # Categorical choices for random states
            'partition_type': ['CPM'],  # Categorical partition types
        },
    resolution_parameters='resolution',  # Mark resolutions as resolution parameter
)

# Sample 10 different parameter combinations
for i in range(10):
    leiden_param = leiden_bootstrapper.sample()

    print(leiden_param)

# Sample 10 different parameter combinations
for i in range(10):
    leiden_param = leiden_bootstrapper.sample(resolution=0.1)

    print(leiden_param)

# # ===================================================================
# # Example 7: Sampling with Specific Resolution Values
# # ===================================================================

# print("\n" + "="*60)
# print("Example 7: Sampling with Specific Resolution Values")  
# print("="*60)

# # Define specific resolution values to use instead of sampling from ranges
# specific_resolutions = {'resolutions': [0.25, 0.5, 1.0, 1.5]}

# # Sample parameters while using only these specific resolutions
# targeted_samples = eval_bootstrapper.sample_parameters(
#     n_samples=8,
#     method='random',
#     resolution_values=specific_resolutions
# )

# print(f"\nGenerated {len(targeted_samples)} samples with specific resolutions:")
# resolution_usage = {}
# for i, params in enumerate(targeted_samples):
#     res = params['resolutions'][0] if isinstance(params['resolutions'], list) else params['resolutions']
#     neighbors = params['n_neighbors']
#     ptype = params['partition_type'] 
    
#     print(f"  Sample {i+1}: res={res}, neighbors={neighbors}, type={ptype}")
#     resolution_usage[res] = resolution_usage.get(res, 0) + 1

# print(f"\nResolution usage distribution:")
# for res, count in sorted(resolution_usage.items()):
#     print(f"  Resolution {res}: used {count} times")

