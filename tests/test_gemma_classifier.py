import pytest
import pandas as pd
import tempfile
import os
from src.gemma_classifier import MockGemmaIrisClassifier


class TestGemmaClassifier:
    """Test Gemma LLM classifier functionality."""
    
    def test_mock_gemma_classifier_initialization(self):
        """Test mock classifier can be initialized."""
        classifier = MockGemmaIrisClassifier()
        assert not classifier.model_loaded
        
        # Load model
        result = classifier.load_model()
        assert result is True
        assert classifier.model_loaded
    
    def test_mock_gemma_prediction(self):
        """Test mock classifier predictions."""
        classifier = MockGemmaIrisClassifier()
        classifier.load_model()
        
        # Test setosa prediction (small petal measurements)
        species, confidence = classifier.predict(5.1, 3.5, 1.4, 0.2)
        assert species == "setosa"
        assert 0.0 <= confidence <= 1.0
        
        # Test versicolor prediction (medium petal measurements)
        species, confidence = classifier.predict(6.0, 3.0, 4.5, 1.5)
        assert species == "versicolor"
        assert 0.0 <= confidence <= 1.0
        
        # Test virginica prediction (large petal measurements)
        species, confidence = classifier.predict(7.0, 3.0, 5.5, 2.0)
        assert species == "virginica"
        assert 0.0 <= confidence <= 1.0
    
    def test_mock_gemma_prediction_without_model_loaded(self):
        """Test that prediction fails if model not loaded."""
        classifier = MockGemmaIrisClassifier()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            classifier.predict(5.1, 3.5, 1.4, 0.2)
    
    def test_mock_gemma_evaluation(self):
        """Test evaluation on test data."""
        # Create test CSV
        test_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 6.0, 7.0],
            'sepal_width': [3.5, 3.0, 3.0, 3.0],
            'petal_length': [1.4, 1.4, 4.5, 5.5],
            'petal_width': [0.2, 0.2, 1.5, 2.0],
            'species': ['setosa', 'setosa', 'versicolor', 'virginica']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            test_csv_path = f.name
        
        try:
            classifier = MockGemmaIrisClassifier()
            classifier.load_model()
            
            # Evaluate
            metrics = classifier.evaluate_on_test_data(test_csv_path)
            
            # Check metrics structure
            assert 'accuracy' in metrics
            assert 'correct_predictions' in metrics
            assert 'total_predictions' in metrics
            assert 'model_type' in metrics
            
            assert metrics['total_predictions'] == len(test_data)
            assert 0.0 <= metrics['accuracy'] <= 1.0
            assert metrics['model_type'] == 'mock_gemma_llm'
            
        finally:
            os.unlink(test_csv_path)
    
    def test_mock_gemma_cleanup(self):
        """Test cleanup functionality."""
        classifier = MockGemmaIrisClassifier()
        classifier.load_model()
        
        # Should not raise any errors
        classifier.cleanup()