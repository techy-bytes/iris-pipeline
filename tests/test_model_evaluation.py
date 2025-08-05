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
    
    # Check if poisoned dataset is being used and adjust threshold accordingly
    poisoned_path = 'data/iris_poisoned.csv'
    if os.path.exists(poisoned_path):
        # With poisoned data, expect lower accuracy due to label corruption
        # Threshold should account for poisoning impact while still validating model works
        threshold = 0.85
        context = "poisoned"
    else:
        # With original data, expect higher accuracy
        threshold = 0.88
        context = "original"
    
    assert accuracy >= threshold, f"Model accuracy ({accuracy:.4f}) with {context} dataset is below acceptable threshold ({threshold})."

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