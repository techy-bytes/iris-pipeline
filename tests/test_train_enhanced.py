import pytest
import pandas as pd
import os
from src.train_enhanced import (
    train_sklearn_model, train_gemma_model_mock, compare_models, 
    train_model, create_metrics_file
)


@pytest.fixture(scope="function", autouse=True)
def cleanup_files():
    """Clean up generated files before and after each test."""
    files_to_clean = ['model.joblib', 'label_encoder.joblib', 'metrics.csv']
    
    # Clean before
    for file in files_to_clean:
        if os.path.exists(file):
            os.remove(file)
    
    yield
    
    # Clean after
    for file in files_to_clean:
        if os.path.exists(file):
            os.remove(file)


class TestEnhancedTraining:
    """Test enhanced training functionality with both models."""
    
    def test_train_sklearn_model(self):
        """Test sklearn model training."""
        accuracy = train_sklearn_model()
        
        # Check accuracy is reasonable
        assert 0.0 <= accuracy <= 1.0
        assert accuracy >= 0.8  # Should be high for iris dataset
        
        # Check files are created
        assert os.path.exists('model.joblib')
        assert os.path.exists('label_encoder.joblib')
    
    def test_train_gemma_model_mock(self):
        """Test mock Gemma model training/evaluation."""
        accuracy = train_gemma_model_mock()
        
        # Check accuracy is reasonable
        assert 0.0 <= accuracy <= 1.0
        assert accuracy >= 0.5  # Mock should perform reasonably
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        results = compare_models()
        
        # Check structure
        assert 'sklearn_accuracy' in results
        assert 'gemma_accuracy' in results
        assert 'winner' in results
        assert 'accuracy_difference' in results
        
        # Check values
        assert 0.0 <= results['sklearn_accuracy'] <= 1.0
        assert 0.0 <= results['gemma_accuracy'] <= 1.0
        assert results['winner'] in ['sklearn', 'gemma', 'tie']
        assert results['accuracy_difference'] >= 0.0
        
        # Check sklearn artifacts are created
        assert os.path.exists('model.joblib')
        assert os.path.exists('label_encoder.joblib')
    
    def test_train_model_sklearn(self):
        """Test training specific model type - sklearn."""
        accuracy = train_model('sklearn')
        assert 0.0 <= accuracy <= 1.0
        assert os.path.exists('model.joblib')
    
    def test_train_model_gemma(self):
        """Test training specific model type - gemma."""
        accuracy = train_model('gemma')
        assert 0.0 <= accuracy <= 1.0
    
    def test_train_model_compare(self):
        """Test training with comparison mode."""
        accuracy = train_model('compare')
        assert 0.0 <= accuracy <= 1.0
    
    def test_train_model_invalid_type(self):
        """Test training with invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            train_model('invalid_model')
    
    def test_create_metrics_file(self):
        """Test metrics file creation."""
        # Create metrics with specific accuracies
        primary_accuracy = create_metrics_file(sklearn_accuracy=0.95, gemma_accuracy=0.90)
        
        assert os.path.exists('metrics.csv')
        assert primary_accuracy == 0.95  # Should return the higher accuracy
        
        # Read and verify metrics file
        metrics_df = pd.read_csv('metrics.csv')
        metrics_dict = dict(zip(metrics_df['metric'], metrics_df['value']))
        
        assert 'accuracy' in metrics_dict
        assert 'sklearn_accuracy' in metrics_dict
        assert 'gemma_accuracy' in metrics_dict
        assert 'best_model' in metrics_dict
        assert 'num_samples' in metrics_dict
        assert 'num_features' in metrics_dict
        assert 'num_classes' in metrics_dict
        
        assert float(metrics_dict['accuracy']) == 0.95
        assert float(metrics_dict['sklearn_accuracy']) == 0.95
        assert float(metrics_dict['gemma_accuracy']) == 0.90
        assert metrics_dict['best_model'] == 'sklearn_rf'
    
    def test_create_metrics_file_with_training(self):
        """Test metrics file creation with automatic training."""
        primary_accuracy = create_metrics_file()
        
        assert os.path.exists('metrics.csv')
        assert 0.0 <= primary_accuracy <= 1.0
        
        # Should have created sklearn artifacts during training
        assert os.path.exists('model.joblib')
        assert os.path.exists('label_encoder.joblib')
        
        # Verify metrics file structure
        metrics_df = pd.read_csv('metrics.csv')
        assert len(metrics_df) >= 7  # At least 7 metrics expected
        
        # Check all required metrics are present
        metric_names = set(metrics_df['metric'].tolist())
        required_metrics = {
            'accuracy', 'sklearn_accuracy', 'gemma_accuracy', 'best_model',
            'num_samples', 'num_features', 'num_classes'
        }
        assert required_metrics.issubset(metric_names)