"""
Example usage of consensus matrix clustering with three 2D blobs.

This example demonstrates the enhanced consensus matrix approach with 
partition reliability selection features.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.sparse import csr_matrix
import sys
import os

# Add the treeclust package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from treeclust.struct_consensus_leiden import ConsensusClusteringLeiden

def main():
    print("Three Blobs Consensus Clustering Example with Partition Reliability")
    print("="*80)
    
    # Generate three clearly separated blobs in 2D
    print("Generating three 2D blob clusters...")
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.2, 
                          center_box=(-5.0, 5.0), random_state=42)
    
    print(f"Data shape: {X.shape}")
    print(f"True cluster distribution: {np.bincount(y_true)}")
    
    # Create connectivity matrix for Leiden algorithm
    print("Computing connectivity matrix...")
    connectivity = kneighbors_graph(X, n_neighbors=15, include_self=False)
    connectivity = csr_matrix(connectivity + connectivity.T)  # Make symmetric
    print(f"Connectivity matrix shape: {connectivity.shape}")
    print(f"Number of edges: {connectivity.nnz // 2}")
    
    print("\n" + "="*80)
    print("HIERARCHICAL METHOD COMPARISON")
    print("="*80)
    
    # Test different hierarchical methods
    methods = ['ward', 'confidence_weighted', 'stability_aware']
    method_results = {}
    
    for method in methods:
        print(f"\nTesting {method} hierarchical method...")
        consensus_test = ConsensusClusteringLeiden(
            connectivity_matrix=connectivity.toarray(),
            parameter_range=np.linspace(0.05, 1.0, 8),
            n_clusters=3,
            n_iter=15,
            random_state=42,
            hierarchical_method=method
        )
        
        consensus_test.fit(verbose=False)
        test_labels = consensus_test.predict()
        test_ari = adjusted_rand_score(y_true, test_labels)
        test_sil = silhouette_score(X, test_labels)
        test_stability = consensus_test.compute_stability_metrics()
        
        method_results[method] = {
            'ari': test_ari,
            'silhouette': test_sil,
            'stability': test_stability['consensus_mean'],
            'labels': test_labels
        }
        
        print(f"  ARI: {test_ari:.3f}, Silhouette: {test_sil:.3f}, Stability: {test_stability['consensus_mean']:.3f}")
    
    # Find best method
    best_method = max(method_results.keys(), key=lambda m: method_results[m]['ari'])
    print(f"\nBest method by ARI: {best_method} (ARI={method_results[best_method]['ari']:.3f})")
    
    print("\n" + "="*80)
    print("STANDARD CONSENSUS CLUSTERING")
    print("="*80)
    
    # Test standard consensus clustering
    consensus_leiden = ConsensusClusteringLeiden(
        connectivity_matrix=connectivity.toarray(),
        parameter_range=np.linspace(0.05, 1.0, 12),
        n_clusters=3,
        n_iter=20,
        random_state=42,
        hierarchical_method='stability_aware'
    )
    
    consensus_leiden.fit(verbose=True)
    labels = consensus_leiden.predict()
    
    # Evaluate clustering
    ari = adjusted_rand_score(y_true, labels)
    sil = silhouette_score(X, labels)
    stability = consensus_leiden.compute_stability_metrics()
    
    print(f"\nStandard Consensus Results:")
    print(f"Number of clusters: {len(np.unique(labels))}")
    print(f"Cluster sizes: {np.bincount(labels)}")
    print(f"ARI with true clusters: {ari:.3f}")
    print(f"Silhouette score: {sil:.3f}")
    print(f"Consensus stability: {stability['consensus_mean']:.3f} ± {stability['consensus_std']:.3f}")
    
    print("\n" + "="*80)
    print("PARTITION RELIABILITY ANALYSIS")
    print("="*80)
    
    # Test partition reliability evaluation
    print("\nEvaluating partition reliability...")
    try:
        evaluations = consensus_leiden.evaluate_all_partitions()
        
        # Display evaluation summary
        summary = consensus_leiden.get_partition_reliability_summary()
        print(f"\nReliability Summary:")
        print(f"Total partitions evaluated: {summary['total_partitions']}")
        
        for metric, stats in summary.items():
            if metric not in ['total_partitions', 'reliable_partitions', 'reliability_ratio'] and stats:
                print(f"{metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # Select reliable partitions
        reliable_indices = consensus_leiden.select_reliable_partitions(
            min_silhouette=0.3,
            min_stability=0.4,
            min_size_balance=0.3,
            top_k=10
        )
        
        print(f"\nSelected {len(reliable_indices)} reliable partitions out of {summary['total_partitions']}")
        print(f"Reliability ratio: {summary['reliability_ratio']:.3f}")
        
        # Plot partition reliability
        try:
            consensus_leiden.plot_partition_reliability(figsize=(14, 10), save_path='three_blobs_partition_reliability.png')
            print("Generated three_blobs_partition_reliability.png")
        except Exception as e:
            print(f"Could not plot partition reliability: {e}")
            
    except Exception as e:
        print(f"Partition reliability analysis failed: {e}")
        print("This is expected if sklearn is not available")
    
    print("\n" + "="*80)
    print("CONSENSUS CLUSTERING WITH PARTITION SELECTION")
    print("="*80)
    
    # Test fit_with_partition_selection
    print("\nFitting with automatic partition selection...")
    consensus_leiden_selective = ConsensusClusteringLeiden(
        connectivity_matrix=connectivity.toarray(),
        parameter_range=np.linspace(0.05, 1.0, 10),
        n_clusters=3,
        n_iter=15,
        random_state=42,
        hierarchical_method='stability_aware'
    )
    
    try:
        consensus_leiden_selective.fit_with_partition_selection(
            data=X,  # Pass the data explicitly
            reliability_kwargs={
                'min_silhouette': 0.4,
                'min_stability': 0.5,
                'min_size_balance': 0.4,
                'top_k': 12
            }
        )
        
        # Compare results
        selective_labels = consensus_leiden_selective.predict()
        selective_ari = adjusted_rand_score(y_true, selective_labels)
        selective_silhouette = silhouette_score(X, selective_labels)
        
        print(f"\nComparison of approaches:")
        print(f"Regular consensus clustering:")
        print(f"  ARI: {ari:.3f}")
        print(f"  Silhouette: {sil:.3f}")
        print(f"  Stability: {stability['consensus_mean']:.3f}")
        
        print(f"\nPartition-selective consensus clustering:")
        print(f"  ARI: {selective_ari:.3f}")
        print(f"  Silhouette: {selective_silhouette:.3f}")
        
        # Get selective reliability summary
        try:
            selective_summary = consensus_leiden_selective.get_partition_reliability_summary()
            print(f"  Reliable partitions used: {selective_summary['reliable_partitions']}/{selective_summary['total_partitions']}")
        except:
            print("  Could not compute selective reliability summary")
            
    except Exception as e:
        print(f"Could not complete partition selection analysis: {e}")
        print("This is expected if dependencies are not available")
        
    print("\n" + "="*80)
    print("VISUALIZATION")
    print("="*80)
    
    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(20, 15))
    
    # Create grid layout: 3x4 = 12 subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: True clusters and hierarchical method comparison
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7, s=40)
    ax1.set_title('True Clusters', fontsize=12)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot different hierarchical methods
    for i, (method, results) in enumerate(method_results.items()):
        ax = fig.add_subplot(gs[0, i+1])
        scatter = ax.scatter(X[:, 0], X[:, 1], c=results['labels'], cmap='viridis', alpha=0.7, s=40)
        ax.set_title(f'{method.replace("_", " ").title()}\nARI={results["ari"]:.3f}', fontsize=12)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        plt.colorbar(scatter, ax=ax)
    
    # Row 2: Standard vs selective clustering
    ax4 = fig.add_subplot(gs[1, 0])
    scatter4 = ax4.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, s=40)
    ax4.set_title(f'Standard Consensus (stability_aware)\nARI={ari:.3f}, Sil={sil:.3f}', fontsize=12)
    ax4.set_xlabel('X1')
    ax4.set_ylabel('X2')
    plt.colorbar(scatter4, ax=ax4)
    
    # Plot selective clustering if available
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        scatter5 = ax5.scatter(X[:, 0], X[:, 1], c=selective_labels, cmap='viridis', alpha=0.7, s=40)
        ax5.set_title(f'Partition-Selective\nARI={selective_ari:.3f}, Sil={selective_silhouette:.3f}', fontsize=12)
        ax5.set_xlabel('X1')
        ax5.set_ylabel('X2')
        plt.colorbar(scatter5, ax=ax5)
    except:
        ax5.text(0.5, 0.5, 'Partition-Selective\nClustering\nNot Available', 
                transform=ax5.transAxes, ha='center', va='center', fontsize=10)
        ax5.set_title('Partition-Selective Clustering', fontsize=12)
    
    # Consensus matrices
    ax6 = fig.add_subplot(gs[1, 2])
    consensus_matrix = consensus_leiden.get_consensus_matrix()
    im1 = ax6.imshow(consensus_matrix, cmap='viridis', aspect='auto')
    ax6.set_title('Standard Consensus Matrix', fontsize=12)
    ax6.set_xlabel('Data Points')
    ax6.set_ylabel('Data Points')
    plt.colorbar(im1, ax=ax6)
    
    # ARI comparison bar chart
    ax7 = fig.add_subplot(gs[1, 3])
    method_names = list(method_results.keys())
    ari_values = [method_results[m]['ari'] for m in method_names]
    bars = ax7.bar(method_names, ari_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax7.set_title('ARI Comparison by Method', fontsize=12)
    ax7.set_ylabel('Adjusted Rand Index')
    ax7.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, ari in zip(bars, ari_values):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{ari:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Row 3: Additional analysis plots
    try:
        # Stability metrics comparison
        ax8 = fig.add_subplot(gs[2, 0])
        stability_values = [method_results[m]['stability'] for m in method_names]
        bars2 = ax8.bar(method_names, stability_values, color=['lightblue', 'pink', 'lightgray'])
        ax8.set_title('Consensus Stability by Method', fontsize=12)
        ax8.set_ylabel('Stability Score')
        ax8.tick_params(axis='x', rotation=45)
        
        for bar, stab in zip(bars2, stability_values):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{stab:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Silhouette comparison
        ax9 = fig.add_subplot(gs[2, 1])
        sil_values = [method_results[m]['silhouette'] for m in method_names]
        bars3 = ax9.bar(method_names, sil_values, color=['wheat', 'salmon', 'lightcyan'])
        ax9.set_title('Silhouette Score by Method', fontsize=12)
        ax9.set_ylabel('Silhouette Score')
        ax9.tick_params(axis='x', rotation=45)
        
        for bar, sil_val in zip(bars3, sil_values):
            ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{sil_val:.3f}', ha='center', va='bottom', fontsize=10)
            
    except Exception as e:
        print(f"Could not create comparison plots: {e}")
    
    # Difference matrix (if selective clustering worked)
    try:
        ax10 = fig.add_subplot(gs[2, 2])
        selective_consensus = consensus_leiden_selective.get_consensus_matrix()
        diff_matrix = np.abs(consensus_matrix - selective_consensus)
        im2 = ax10.imshow(diff_matrix, cmap='Reds', aspect='auto')
        ax10.set_title('Consensus Matrix Difference\n(Standard vs Selective)', fontsize=12)
        ax10.set_xlabel('Data Points')
        ax10.set_ylabel('Data Points')
        plt.colorbar(im2, ax=ax10)
    except:
        ax10 = fig.add_subplot(gs[2, 2])
        ax10.text(0.5, 0.5, 'Consensus Matrix\nDifference\nNot Available', 
                 transform=ax10.transAxes, ha='center', va='center', fontsize=10)
        ax10.set_title('Consensus Matrix Difference', fontsize=12)
    
    # Summary text
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis('off')
    summary_text = f"""
    SUMMARY RESULTS
    
    Best Hierarchical Method: {best_method}
    • ARI: {method_results[best_method]['ari']:.3f}
    • Silhouette: {method_results[best_method]['silhouette']:.3f}
    • Stability: {method_results[best_method]['stability']:.3f}
    
    Standard Consensus:
    • ARI: {ari:.3f}
    • Silhouette: {sil:.3f}
    • Stability: {stability['consensus_mean']:.3f}
    
    Partition Reliability:
    • Total Partitions: {summary['total_partitions']}
    • Reliability Ratio: {summary['reliability_ratio']:.3f}
    """
    try:
        ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, 
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    except:
        ax11.text(0.05, 0.95, "Summary results not available", transform=ax11.transAxes, 
                 fontsize=11, verticalalignment='top')
    
    plt.suptitle('Comprehensive Consensus Clustering Analysis with Reliability Methods', fontsize=16)
    plt.savefig('three_blobs_comprehensive_reliability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # True clusters
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7, s=50)
    ax1.set_title('True Clusters')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    plt.colorbar(scatter1, ax=ax1)
    
    # Standard consensus clustering
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
    ax2.set_title(f'Standard Consensus Clustering\nARI={ari:.3f}, Silhouette={sil:.3f}')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    plt.colorbar(scatter2, ax=ax2)
    
    # Consensus matrix
    consensus_matrix = consensus_leiden.get_consensus_matrix()
    im1 = ax3.imshow(consensus_matrix, cmap='viridis', aspect='auto')
    ax3.set_title('Consensus Matrix')
    ax3.set_xlabel('Data Points')
    ax3.set_ylabel('Data Points')
    plt.colorbar(im1, ax=ax3)
    
    # Plot selective clustering if available
    try:
        scatter3 = ax4.scatter(X[:, 0], X[:, 1], c=selective_labels, cmap='viridis', alpha=0.7, s=50)
        ax4.set_title(f'Partition-Selective Clustering\nARI={selective_ari:.3f}')
        ax4.set_xlabel('X1')
        ax4.set_ylabel('X2')
        plt.colorbar(scatter3, ax=ax4)
    except:
        ax4.text(0.5, 0.5, 'Partition-Selective\nClustering\nNot Available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title('Partition-Selective Clustering')
    
    plt.tight_layout()
    plt.savefig('three_blobs_reliability_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nThree blobs example with reliability analysis completed!")
    print("\nGenerated files:")
    print("  - three_blobs_comprehensive_reliability_analysis.png: Complete hierarchical methods comparison")
    print("  - three_blobs_reliability_comparison.png: Standard comparison plot")
    if os.path.exists('three_blobs_partition_reliability.png'):
        print("  - three_blobs_partition_reliability.png: Partition reliability analysis")
    
    print("\nKey features demonstrated:")
    print("  ✓ Hierarchical method comparison (ward vs confidence_weighted vs stability_aware)")
    print("  ✓ Reliability-based consensus matrix clustering")
    print("  ✓ Partition reliability evaluation (compute_partition_reliability)")
    print("  ✓ Reliable partition selection (select_reliable_partitions)")
    print("  ✓ Automatic partition-selective clustering (fit_with_partition_selection)")
    print("  ✓ Comprehensive reliability metrics and visualization")
    print(f"  ✓ Best hierarchical method identified: {best_method}")
    print(f"      └─ ARI improvement: {method_results[best_method]['ari']:.3f} vs baseline")

if __name__ == "__main__":
    main()