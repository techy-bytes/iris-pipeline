"""Tests to compare LogisticRegression vs RandomForest for cost-efficiency."""

import pytest
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Global variables to store metrics between tests
_lr_metrics = None
_rf_metrics = None


@pytest.fixture
def iris_data():
    """Load and prepare iris data."""
    df = pd.read_csv('data/iris.csv')
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]
    y = df['species']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, le


def test_logistic_regression_performance(iris_data):
    """Test LogisticRegression performance and efficiency."""
    X_train, X_test, y_train, y_test, le = iris_data
    
    # Train LogisticRegression
    start_time = time.time()
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_training_time = time.time() - start_time
    
    # Test prediction time
    start_time = time.time()
    lr_predictions = lr_model.predict(X_test)
    lr_prediction_time = time.time() - start_time
    
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    
    # Assertions for LogisticRegression
    assert lr_accuracy >= 0.85, f"LogisticRegression accuracy ({lr_accuracy:.4f}) below acceptable threshold"
    assert lr_training_time < 1.0, f"LogisticRegression training too slow ({lr_training_time:.4f}s)"
    assert lr_prediction_time < 0.1, f"LogisticRegression prediction too slow ({lr_prediction_time:.4f}s)"
    
    # Store metrics for comparison (store in global for access by other tests)
    global _lr_metrics
    _lr_metrics = {
        'model': 'LogisticRegression',
        'accuracy': lr_accuracy,
        'training_time': lr_training_time,
        'prediction_time': lr_prediction_time,
        'num_parameters': len(lr_model.coef_[0]) + len(lr_model.classes_)  # weights + bias per class
    }


def test_random_forest_comparison(iris_data):
    """Test RandomForest for comparison with LogisticRegression."""
    X_train, X_test, y_train, y_test, le = iris_data
    
    # Train RandomForest
    start_time = time.time()
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_training_time = time.time() - start_time
    
    # Test prediction time
    start_time = time.time()
    rf_predictions = rf_model.predict(X_test)
    rf_prediction_time = time.time() - start_time
    
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    
    # RandomForest metrics (store in global for access by other tests)
    global _rf_metrics
    _rf_metrics = {
        'model': 'RandomForest',
        'accuracy': rf_accuracy,
        'training_time': rf_training_time,
        'prediction_time': rf_prediction_time,
        'num_parameters': rf_model.n_estimators * sum(tree.tree_.node_count for tree in rf_model.estimators_)
    }


def test_cost_efficiency_comparison(iris_data):
    """Compare LogisticRegression vs RandomForest for cost efficiency."""
    # Ensure the other tests have run to populate the global metrics
    test_logistic_regression_performance(iris_data)
    test_random_forest_comparison(iris_data)
    
    # Get metrics from global variables
    lr_metrics = _lr_metrics
    rf_metrics = _rf_metrics
    
    # Compare cost efficiency metrics
    assert lr_metrics['training_time'] < rf_metrics['training_time'], \
        f"LogisticRegression should be faster to train: LR={lr_metrics['training_time']:.4f}s vs RF={rf_metrics['training_time']:.4f}s"
    
    assert lr_metrics['prediction_time'] <= rf_metrics['prediction_time'], \
        f"LogisticRegression should be faster at prediction: LR={lr_metrics['prediction_time']:.4f}s vs RF={rf_metrics['prediction_time']:.4f}s"
    
    assert lr_metrics['num_parameters'] < rf_metrics['num_parameters'], \
        f"LogisticRegression should have fewer parameters: LR={lr_metrics['num_parameters']} vs RF={rf_metrics['num_parameters']}"
    
    # Accuracy should be competitive (within 5% is acceptable for cost savings)
    accuracy_diff = abs(lr_metrics['accuracy'] - rf_metrics['accuracy'])
    assert accuracy_diff <= 0.05, \
        f"Accuracy difference too large: LR={lr_metrics['accuracy']:.4f} vs RF={rf_metrics['accuracy']:.4f}"
    
    # Create comparison metrics file for CI/CD pipeline
    comparison_metrics = {
        'logistic_regression': lr_metrics,
        'random_forest': rf_metrics,
        'efficiency_improvement': {
            'training_speedup': rf_metrics['training_time'] / lr_metrics['training_time'],
            'prediction_speedup': rf_metrics['prediction_time'] / lr_metrics['prediction_time'],
            'parameter_reduction': (rf_metrics['num_parameters'] - lr_metrics['num_parameters']) / rf_metrics['num_parameters'] * 100,
            'accuracy_retention': lr_metrics['accuracy'] / rf_metrics['accuracy'] * 100
        }
    }
    
    # Write comparison metrics to CSV for workflow results
    comparison_df = pd.DataFrame([
        ['lr_accuracy', lr_metrics['accuracy']],
        ['rf_accuracy', rf_metrics['accuracy']],
        ['lr_training_time', lr_metrics['training_time']],
        ['rf_training_time', rf_metrics['training_time']],
        ['lr_prediction_time', lr_metrics['prediction_time']],
        ['rf_prediction_time', rf_metrics['prediction_time']],
        ['lr_parameters', lr_metrics['num_parameters']],
        ['rf_parameters', rf_metrics['num_parameters']],
        ['training_speedup', comparison_metrics['efficiency_improvement']['training_speedup']],
        ['prediction_speedup', comparison_metrics['efficiency_improvement']['prediction_speedup']],
        ['parameter_reduction_percent', comparison_metrics['efficiency_improvement']['parameter_reduction']],
        ['accuracy_retention_percent', comparison_metrics['efficiency_improvement']['accuracy_retention']]
    ], columns=['metric', 'value'])
    
    comparison_df.to_csv('model_comparison_metrics.csv', index=False)
    
    print(f"\nModel Comparison Results:")
    print(f"LogisticRegression accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"RandomForest accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"Training speedup: {comparison_metrics['efficiency_improvement']['training_speedup']:.2f}x")
    print(f"Prediction speedup: {comparison_metrics['efficiency_improvement']['prediction_speedup']:.2f}x")
    print(f"Parameter reduction: {comparison_metrics['efficiency_improvement']['parameter_reduction']:.1f}%")
    print(f"Accuracy retention: {comparison_metrics['efficiency_improvement']['accuracy_retention']:.1f}%")


def test_current_model_is_logistic_regression():
    """Verify that the current trained model is LogisticRegression."""
    from src.train import train_model
    
    # Train the model
    train_model()
    
    # Load the saved model
    model = joblib.load('model.joblib')
    
    # Verify it's LogisticRegression
    assert isinstance(model, LogisticRegression), \
        f"Expected LogisticRegression, got {type(model).__name__}"
    
    print(f"✓ Current model is LogisticRegression")


@pytest.fixture(autouse=True)
def cleanup_comparison_files():
    """Clean up comparison files before and after tests."""
    files_to_cleanup = ['model_comparison_metrics.csv']
    
    # Clean up before test
    for file in files_to_cleanup:
        if os.path.exists(file):
            os.remove(file)
    
    yield
    
    # Keep the comparison metrics file for CI/CD pipeline to use
    # Don't clean up model_comparison_metrics.csv after tests