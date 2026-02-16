#!/usr/bin/env python3
"""
Example demonstrating classifier sampling functionality in ClusteringClassBootstrapper.

This example shows how to sample different classifiers with varying parameters,
which is useful for ensemble methods and robust classification tasks.
"""

import numpy as np
import sys
import os

# Add the treeclust directory to Python path
sys.path.insert(0, '/Users/gatocor/Documents/Academic/UPF Postdoc/treeclust')

from treeclust import ClusteringClassBootstrapper, DataBootstrapper

def test_classifier_sampling():
    """Demonstrate classifier sampling functionality."""
    
    print("Classifier Sampling Example")
    print("="*50)
    
    # Create mock classifiers since sklearn might not be available
    class MockKNeighborsClassifier:
        def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto'):
            self.n_neighbors = n_neighbors
            self.weights = weights
            self.algorithm = algorithm
        
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            return np.zeros(len(X))
        
        def get_params(self):
            return {
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'algorithm': self.algorithm
            }
    
    class MockRandomForestClassifier:
        def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=None):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.random_state = random_state
        
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            return np.zeros(len(X))
        
        def get_params(self):
            return {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'random_state': self.random_state
            }
    
    class MockSVM:
        def __init__(self, C=1.0, kernel='rbf', gamma='scale', random_state=None):
            self.C = C
            self.kernel = kernel
            self.gamma = gamma
            self.random_state = random_state
        
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            return np.zeros(len(X))
        
        def get_params(self):
            return {
                'C': self.C,
                'kernel': self.kernel,
                'gamma': self.gamma,
                'random_state': self.random_state
            }
    
    # ===================================================================
    # Example 1: Basic Classifier Sampling
    # ===================================================================
    
    print("\n1. Basic Classifier Sampling")
    print("-" * 30)
    
    # Create a basic parameter bootstrapper (even though we won't use it for parameter sampling)
    bootstrapper = ClusteringClassBootstrapper(
        ml_class=object,  # Dummy class
        parameter_ranges={},
        random_state=42
    )
    
    # Define classifier choices and their parameter ranges
    classifier_choices = {
        'knn': MockKNeighborsClassifier,
        'rf': MockRandomForestClassifier,
        'svm': MockSVM
    }
    
    classifier_parameters = {
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        },
        'rf': {
            'n_estimators': [10, 25, 50, 100, 200],
            'max_depth': (3, 15),  # Continuous range
            'min_samples_split': [2, 5, 10],
            'random_state': [42]
        },
        'svm': {
            'C': (0.1, 10.0),  # Continuous range
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto'],
            'random_state': [42]
        }
    }
    
    # Sample multiple classifiers
    print("Sampling 10 classifiers with random parameters:")
    sampled_classifiers = []
    
    for i in range(10):
        classifier = bootstrapper.sample_classifier(classifier_choices, classifier_parameters)
        class_name = type(classifier).__name__
        params = classifier.get_params()
        
        # Determine classifier type
        classifier_type = None
        for name, cls in classifier_choices.items():
            if isinstance(classifier, cls):
                classifier_type = name
                break
        
        sampled_classifiers.append({
            'instance': classifier,
            'type': classifier_type,
            'class_name': class_name,
            'parameters': params
        })
        
        print(f"  {i+1:2d}. {classifier_type:3s} -> {class_name}")
        # Show a few key parameters
        key_params = {}
        if classifier_type == 'knn':
            key_params = {'n_neighbors': params.get('n_neighbors'), 'weights': params.get('weights')}
        elif classifier_type == 'rf':
            key_params = {'n_estimators': params.get('n_estimators'), 'max_depth': params.get('max_depth')}
        elif classifier_type == 'svm':
            key_params = {'C': f"{params.get('C'):.2f}" if params.get('C') else None, 'kernel': params.get('kernel')}
        print(f"      {key_params}")
    
    # ===================================================================
    # Example 2: Classifier Sampling with Config
    # ===================================================================
    
    print("\n\n2. Classifier Sampling with Configuration Metadata")
    print("-" * 50)
    
    config = {
        'choices': classifier_choices,
        'parameters': classifier_parameters
    }
    
    print("Sampling 5 classifiers with full metadata:")
    for i in range(5):
        result = bootstrapper.sample_classifier_with_config(config)
        
        print(f"  Sample {i+1}:")
        print(f"    Type: {result['name']}")
        print(f"    Class: {result['class_name']}")
        print(f"    Parameters: {result['parameters']}")
        print()
    
    # ===================================================================
    # Example 3: Integration with Data DataBootstrapper
    # ===================================================================
    
    print("\n3. Integration with Data DataBootstrapper")
    print("-" * 40)
    
    # Generate some mock data
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 3, 100)
    
    print(f"Generated mock data: X shape {X.shape}, y shape {y.shape}")
    
    # Create bootstrapping instances with different sampled classifiers
    print("\nCreating DataBootstrapper instances with sampled classifiers:")
    
    for i in range(3):
        # Sample a classifier
        classifier = bootstrapper.sample_classifier(classifier_choices, classifier_parameters)
        classifier_type = None
        for name, cls in classifier_choices.items():
            if isinstance(classifier, cls):
                classifier_type = name
                break
        
        # Create bootstrapping instance
        bootstrap = DataBootstrapper(
            sample_ratio=0.7,
            classifier=classifier,
            random_state=i
        )
        
        print(f"  Bootstrap {i+1}: {classifier_type} classifier")
        print(f"    Sample ratio: {bootstrap.sample_ratio}")
        print(f"    Classifier: {type(bootstrap.classifier).__name__}")
        
        # Test the bootstrap functionality
        try:
            X_train, X_test, train_idx, test_idx = bootstrap.bootstrap_sample(X)
            y_train = y[train_idx]
            
            # Test prediction (will return zeros from mock classifier)
            y_pred = bootstrap.predict_excluded(X_train, X_test, y_train)
            
            print(f"    Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            print(f"    Predictions shape: {y_pred.shape}")
        except Exception as e:
            print(f"    Bootstrap test failed: {e}")
        
        print()
    
    # ===================================================================
    # Example 4: Classifier Distribution Analysis
    # ===================================================================
    
    print("\n4. Classifier Distribution Analysis")
    print("-" * 35)
    
    # Sample many classifiers to see distribution
    classifier_counts = {}
    parameter_stats = {}
    
    n_samples = 100
    print(f"Sampling {n_samples} classifiers to analyze distribution:")
    
    for _ in range(n_samples):
        classifier = bootstrapper.sample_classifier(classifier_choices, classifier_parameters)
        
        # Count classifier types
        classifier_type = None
        for name, cls in classifier_choices.items():
            if isinstance(classifier, cls):
                classifier_type = name
                break
        
        classifier_counts[classifier_type] = classifier_counts.get(classifier_type, 0) + 1
        
        # Collect parameter statistics
        params = classifier.get_params()
        if classifier_type not in parameter_stats:
            parameter_stats[classifier_type] = {}
        
        for param_name, param_value in params.items():
            if param_name not in parameter_stats[classifier_type]:
                parameter_stats[classifier_type][param_name] = []
            parameter_stats[classifier_type][param_name].append(param_value)
    
    # Display results
    print("\nClassifier type distribution:")
    for classifier_type, count in sorted(classifier_counts.items()):
        percentage = (count / n_samples) * 100
        print(f"  {classifier_type:3s}: {count:3d} samples ({percentage:5.1f}%)")
    
    print("\nParameter statistics (for numeric parameters):")
    for classifier_type, params in parameter_stats.items():
        print(f"\n  {classifier_type.upper()}:")
        for param_name, values in params.items():
            # Only show stats for numeric parameters
            try:
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    mean_val = np.mean(numeric_values)
                    std_val = np.std(numeric_values)
                    min_val = np.min(numeric_values)
                    max_val = np.max(numeric_values)
                    print(f"    {param_name}: mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
            except:
                pass
    
    print("\n✅ Classifier sampling examples completed successfully!")
    
    print("\n" + "="*60)
    print("SUMMARY: Classifier Sampling Features")
    print("="*60)
    print("✓ Random classifier type selection from user-defined choices")
    print("✓ Parameter sampling with continuous, discrete, and categorical ranges")
    print("✓ Intelligent parameter handling (e.g., integer conversion)")
    print("✓ Fallback handling for failed instantiations") 
    print("✓ Metadata extraction and configuration tracking")
    print("✓ Integration with data bootstrapping workflows")
    print("✓ Distribution analysis and parameter statistics")
    print("✓ Extensible to any classifier with fit/predict interface")

if __name__ == "__main__":
    try:
        test_classifier_sampling()
    except Exception as e:
        print(f"\n💥 Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)