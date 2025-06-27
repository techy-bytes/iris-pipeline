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
    # The Iris dataset is relatively easy, so a high accuracy is expected.
    assert accuracy >= 0.95, f"Model accuracy ({accuracy:.4f}) is below acceptable threshold (0.95)."

def test_metrics_file_content():
    """Verify the content and structure of the metrics.csv file."""
    train_model()
    assert os.path.exists('metrics.csv'), "metrics.csv was not created."
    
    metrics_df = pd.read_csv('metrics.csv')
    
    assert list(metrics_df.columns) == ['metric', 'value'], "metrics.csv has unexpected columns."
    
    metrics_dict = dict(zip(metrics_df['metric'], metrics_df['value']))
    
    assert 'accuracy' in metrics_dict and isinstance(metrics_dict['accuracy'], float), "Accuracy metric missing or incorrect type."
    assert 0.0 <= metrics_dict['accuracy'] <= 1.0, "Accuracy value is out of valid range [0, 1]."
    assert 'num_samples' in metrics_dict and isinstance(metrics_dict['num_samples'], int) and metrics_dict['num_samples'] == 150, "num_samples metric missing or incorrect."
    assert 'num_features' in metrics_dict and isinstance(metrics_dict['num_features'], int) and metrics_dict['num_features'] == 4, "num_features metric missing or incorrect."
    assert 'num_classes' in metrics_dict and isinstance(metrics_dict['num_classes'], int) and metrics_dict['num_classes'] == 3, "num_classes metric missing or incorrect."