import pytest
import pandas as pd
import os
from src.train import train_model # Import the function directly

# Fixture to ensure metrics.csv is cleaned up after tests
@pytest.fixture(scope="function", autouse=True)
def cleanup_metrics_file():
    """Ensures metrics.csv is removed before and after each test."""
    metrics_path = 'metrics.csv'
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    yield
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

def test_model_training_produces_metrics_file():
    """Verify that the train_model function creates metrics.csv."""
    train_model()
    assert os.path.exists('metrics.csv'), "metrics.csv was not created after training."

def test_model_accuracy_threshold():
    """Verify that the trained model's accuracy meets an acceptable threshold."""
    accuracy = train_model()
    
    # Check if poisoned dataset is being used - fail the build if poisoning is detected
    poisoned_path = 'data/iris_poisoned.csv'
    if os.path.exists(poisoned_path):
        pytest.fail(f"DATA POISONING DETECTED: Poisoned dataset found at {poisoned_path}. "
                   f"Build failed for security reasons. Model accuracy with poisoned data: {accuracy:.4f}")
    
    # With original data, expect high accuracy
    threshold = 0.88
    assert accuracy >= threshold, f"Model accuracy ({accuracy:.4f}) is below acceptable threshold ({threshold})."

def test_metrics_file_content():
    """Verify the content and structure of the metrics.csv file."""
    train_model()
    assert os.path.exists('metrics.csv'), "metrics.csv was not created."
    
    metrics_df = pd.read_csv('metrics.csv')
    
    assert list(metrics_df.columns) == ['metric', 'value'], "metrics.csv has unexpected columns."
    
    metrics_dict = dict(zip(metrics_df['metric'], metrics_df['value']))
    
    assert 'accuracy' in metrics_dict and isinstance(metrics_dict['accuracy'], float), "Accuracy metric missing or incorrect type."
    assert 0.0 <= metrics_dict['accuracy'] <= 1.0, "Accuracy value is out of valid range [0, 1]."
    # Pandas reads all numeric values from the CSV as floats, so we cast to int for comparison.
    assert 'num_samples' in metrics_dict and int(metrics_dict['num_samples']) == 150, "num_samples metric missing or incorrect."
    assert 'num_features' in metrics_dict and int(metrics_dict['num_features']) == 4, "num_features metric missing or incorrect."
    assert 'num_classes' in metrics_dict and int(metrics_dict['num_classes']) == 3, "num_classes metric missing or incorrect."

def test_no_data_poisoning_detected():
    """Critical security test: Fail build if data poisoning is detected."""
    original_path = 'data/iris.csv'
    poisoned_path = 'data/iris_poisoned.csv'
    
    # Ensure original data exists
    assert os.path.exists(original_path), f"Original dataset not found at {original_path}"
    
    # Fail immediately if poisoned dataset is detected
    if os.path.exists(poisoned_path):
        # Load both datasets to provide detailed poisoning information
        original_data = pd.read_csv(original_path)
        poisoned_data = pd.read_csv(poisoned_path)
        
        # Calculate poisoning statistics
        different_labels = (original_data['species'] != poisoned_data['species']).sum()
        total_samples = len(original_data)
        poison_rate = different_labels / total_samples
        
        # Get distribution changes
        orig_dist = original_data['species'].value_counts().to_dict()
        poison_dist = poisoned_data['species'].value_counts().to_dict()
        
        # Create detailed failure message
        error_msg = (
            f"🚨 CRITICAL SECURITY ALERT: DATA POISONING DETECTED 🚨\n"
            f"Poisoned dataset found at: {poisoned_path}\n"
            f"Poisoning rate: {poison_rate:.1%} ({different_labels}/{total_samples} samples)\n"
            f"Original distribution: {orig_dist}\n"
            f"Poisoned distribution: {poison_dist}\n"
            f"BUILD FAILED for security reasons. Remove poisoned dataset to proceed."
        )
        
        pytest.fail(error_msg)
    
    print("✅ No data poisoning detected - build can proceed safely")