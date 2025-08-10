import pytest
import pandas as pd
import numpy as np
import os
import json
from src.train import add_location_attribute, compute_fairness_metrics, generate_feature_importance_analysis

def test_add_location_attribute():
    """Test that location attribute is added correctly."""
    # Create a small test dataframe
    test_df = pd.DataFrame({
        'sepal_length': [5.1, 4.9, 4.7],
        'sepal_width': [3.5, 3.0, 3.2],
        'petal_length': [1.4, 1.4, 1.3],
        'petal_width': [0.2, 0.2, 0.2],
        'species': ['setosa', 'setosa', 'setosa']
    })
    
    original_cols = len(test_df.columns)
    result_df = add_location_attribute(test_df.copy())  # Use copy to avoid modifying original
    
    # Check that location column is added
    assert 'location' in result_df.columns, "Location column should be added"
    
    # Check that location values are 0 or 1
    assert all(val in [0, 1] for val in result_df['location']), "Location values should be 0 or 1"
    
    # Check that original data is preserved
    assert len(result_df) == len(test_df), "Number of rows should be preserved"
    assert len(result_df.columns) == original_cols + 1, "Should add exactly one column"

def test_enhanced_dataset_creation():
    """Test that enhanced dataset with location is created during training."""
    from src.train import train_model
    
    # Remove existing files to ensure clean test
    if os.path.exists('data/iris_with_location.csv'):
        os.remove('data/iris_with_location.csv')
    if os.path.exists('fairness_analysis.json'):
        os.remove('fairness_analysis.json')
    
    # Run training
    accuracy = train_model()
    
    # Check that enhanced dataset is created
    assert os.path.exists('data/iris_with_location.csv'), "Enhanced dataset should be created"
    
    # Load and verify enhanced dataset
    enhanced_df = pd.read_csv('data/iris_with_location.csv')
    assert 'location' in enhanced_df.columns, "Enhanced dataset should have location column"
    assert enhanced_df.shape[0] == 150, "Enhanced dataset should have 150 rows"
    assert enhanced_df.shape[1] == 6, "Enhanced dataset should have 6 columns"
    
    # Verify location values
    assert set(enhanced_df['location'].unique()) == {0, 1}, "Location should only have values 0 and 1"

def test_fairness_analysis_creation():
    """Test that fairness analysis is generated correctly."""
    from src.train import train_model
    
    # Remove existing file to ensure clean test
    if os.path.exists('fairness_analysis.json'):
        os.remove('fairness_analysis.json')
    
    # Run training
    train_model()
    
    # Check that fairness analysis file is created
    assert os.path.exists('fairness_analysis.json'), "Fairness analysis file should be created"
    
    # Load and verify fairness analysis
    with open('fairness_analysis.json', 'r') as f:
        analysis = json.load(f)
    
    # Check structure
    assert 'fairness_metrics' in analysis, "Should contain fairness metrics"
    assert 'feature_analysis' in analysis, "Should contain feature analysis"
    
    fairness_metrics = analysis['fairness_metrics']
    
    # Check required fairness metrics
    required_metrics = [
        'overall_accuracy', 'accuracy_location_0', 'accuracy_location_1', 
        'accuracy_difference', 'prediction_rate_virginica_location_0',
        'prediction_rate_virginica_location_1', 'prediction_rate_diff_virginica'
    ]
    
    for metric in required_metrics:
        assert metric in fairness_metrics, f"Should contain {metric}"
    
    # Check that metrics are reasonable
    assert 0 <= fairness_metrics['overall_accuracy'] <= 1, "Overall accuracy should be between 0 and 1"
    assert 0 <= fairness_metrics['accuracy_location_0'] <= 1, "Location 0 accuracy should be between 0 and 1"
    assert 0 <= fairness_metrics['accuracy_location_1'] <= 1, "Location 1 accuracy should be between 0 and 1"

def test_feature_importance_analysis():
    """Test that feature importance analysis is generated correctly."""
    from src.train import train_model
    
    # Run training
    train_model()
    
    # Load fairness analysis
    with open('fairness_analysis.json', 'r') as f:
        analysis = json.load(f)
    
    feature_analysis = analysis['feature_analysis']
    
    # Check structure
    assert 'feature_importances' in feature_analysis, "Should contain feature importances"
    assert 'top_features' in feature_analysis, "Should contain top features"
    assert 'virginica_analysis' in feature_analysis, "Should contain virginica analysis"
    
    # Check feature importances
    importances = feature_analysis['feature_importances']
    expected_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    for feature in expected_features:
        assert feature in importances, f"Should contain importance for {feature}"
        assert 0 <= importances[feature] <= 1, f"Importance for {feature} should be between 0 and 1"
    
    # Check that importances sum to approximately 1
    total_importance = sum(importances.values())
    assert abs(total_importance - 1.0) < 0.01, "Feature importances should sum to approximately 1"
    
    # Check virginica analysis
    virginica_analysis = feature_analysis['virginica_analysis']
    assert 'description' in virginica_analysis, "Should contain description"
    assert 'key_insights' in virginica_analysis, "Should contain key insights"
    assert len(virginica_analysis['key_insights']) > 0, "Should have at least one insight"

def test_shap_explanation_file_exists():
    """Test that SHAP explanation file is created."""
    assert os.path.exists('SHAP_Virginica_Explanation.md'), "SHAP explanation file should exist"
    
    # Check that file has content
    with open('SHAP_Virginica_Explanation.md', 'r') as f:
        content = f.read()
    
    assert len(content) > 0, "SHAP explanation file should not be empty"
    assert 'virginica' in content.lower(), "Should mention virginica"
    assert 'shap' in content.lower(), "Should mention SHAP"
    assert 'feature' in content.lower(), "Should mention features"

# Cleanup fixture
@pytest.fixture(scope="function", autouse=True)
def cleanup_fairness_files():
    """Clean up generated fairness files after tests."""
    yield
    # Cleanup files generated during tests
    for file in ['fairness_analysis.json', 'data/iris_with_location.csv']:
        if os.path.exists(file):
            # Don't remove them as they're part of the feature
            pass