"""
Simple test of the ConsensusCluster class functionality.
"""

import numpy as np
from treeclust import ConsensusCluster

# Create simple test partitions
print("Creating test partitions...")

# Simple partitions for 8 samples
partitions = {
    'partition_1': [0, 0, 1, 1, 2, 2, 2, 1],
    'partition_2': [0, 0, 1, 1, 2, 2, 2, 2], 
    'partition_3': [0, 0, 0, 1, 1, 2, 2, 2],
    'partition_4': [0, 0, 1, 1, 1, 2, 2, 2],
    'partition_5': [0, 1, 1, 1, 2, 2, 2, 0]
}

print(f"Test partitions:")
for key, partition in partitions.items():
    print(f"  {key}: {partition}")

# Create and fit consensus cluster
print(f"\nTesting ConsensusCluster...")
consensus = ConsensusCluster(linkage_method='complete')

print("Fitting consensus model...")
consensus.fit(partitions)

print(f"Consensus model: {consensus}")

# Get coassociation matrix
print(f"\nCoassociation matrix:")
coassoc_matrix = consensus.get_coassociation_matrix('basic')
print(coassoc_matrix)

print(f"\nMatrix statistics:")
print(f"  Shape: {coassoc_matrix.shape}")
print(f"  Mean: {np.mean(coassoc_matrix):.3f}")
print(f"  Min: {np.min(coassoc_matrix):.3f}")
print(f"  Max: {np.max(coassoc_matrix):.3f}")

# Get distance matrix
print(f"\nDistance matrix (1 - coassociation):")
distance_matrix = consensus.get_distance_matrix('basic')
print(distance_matrix)

# Get consensus partitions
print(f"\nConsensus partitions:")
for k in [2, 3, 4]:
    try:
        labels = consensus.predict(k)
        print(f"  K={k}: {labels}")
    except Exception as e:
        print(f"  K={k}: Error - {e}")

# Test stability metrics
print(f"\nStability metrics:")
try:
    metrics = consensus.compute_stability_metrics()
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
except Exception as e:
    print(f"  Error: {e}")

print(f"\nTest completed successfully!")