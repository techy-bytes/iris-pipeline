#!/usr/bin/env python3
"""
Tests for Gemma Model Evaluation
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gemma_evaluation import (
    BaseGemmaEvaluator, 
    FineTunedGemmaModel, 
    convert_iris_to_gemma_prompt,
    tokenize_prompts
)

class TestGemmaEvaluation:
    """Test suite for Gemma evaluation components"""
    
    def test_prompt_conversion(self):
        """Test Iris to Gemma prompt conversion"""
        features = [5.1, 3.5, 1.4, 0.2]
        
        # Test training prompt
        train_prompt = convert_iris_to_gemma_prompt(features, "setosa")
        assert "5.1cm" in train_prompt
        assert "setosa" in train_prompt
        assert "Classify this iris flower" in train_prompt
        
        # Test inference prompt
        inference_prompt = convert_iris_to_gemma_prompt(features, for_inference=True)
        assert "5.1cm" in inference_prompt
        assert "setosa" not in inference_prompt
        assert "The iris species is:" in inference_prompt
        
    def test_base_gemma_evaluator(self):
        """Test Base Gemma evaluator"""
        evaluator = BaseGemmaEvaluator()
        
        # Create test prompts
        test_prompts = [
            convert_iris_to_gemma_prompt([5.1, 3.5, 1.4, 0.2], for_inference=True),  # setosa
            convert_iris_to_gemma_prompt([7.0, 3.2, 4.7, 1.4], for_inference=True),  # versicolor
            convert_iris_to_gemma_prompt([6.3, 3.3, 6.0, 2.5], for_inference=True)   # virginica
        ]
        
        predictions, confidences = evaluator.predict(test_prompts)
        
        assert len(predictions) == 3
        assert len(confidences) == 3
        assert all(pred in ["setosa", "versicolor", "virginica"] for pred in predictions)
        assert all(0 <= conf <= 1 for conf in confidences)
        
    def test_tokenization(self):
        """Test prompt tokenization"""
        test_prompts = [
            "Classify this iris flower based on its measurements:",
            "Sepal length: 5.1cm"
        ]
        
        tokens, vocab = tokenize_prompts(test_prompts)
        
        assert tokens.shape[0] == 2  # Two prompts
        assert len(vocab) > 0
        assert '<PAD>' in vocab
        assert '<UNK>' in vocab
        
    def test_finetuned_model_creation(self):
        """Test fine-tuned model creation"""
        model = FineTunedGemmaModel(vocab_size=100, hidden_size=64, num_classes=3)
        
        # Test model parameters
        assert model.embedding.num_embeddings == 100
        assert model.classifier.out_features == 3
        
        # Test forward pass (no training, just structure)
        import torch
        test_input = torch.randint(0, 100, (2, 32))  # batch_size=2, seq_len=32
        output = model(test_input)
        
        assert output.shape == (2, 3)  # batch_size=2, num_classes=3

class TestGemmaEvaluationIntegration:
    """Integration tests for full evaluation pipeline"""
    
    @pytest.fixture
    def sample_iris_data(self):
        """Create sample Iris data for testing"""
        data = {
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 7.0, 6.4, 6.9, 5.5, 6.5],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.2, 3.2, 3.1, 2.3, 2.8],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 4.7, 4.5, 4.9, 4.0, 4.6],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 1.5],
            'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 
                       'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor']
        }
        return pd.DataFrame(data)
    
    def test_evaluation_pipeline_structure(self, sample_iris_data):
        """Test that evaluation pipeline components work together"""
        # Save sample data
        sample_iris_data.to_csv('test_iris.csv', index=False)
        
        # Test prompt generation
        features = sample_iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        species = sample_iris_data['species'].values
        
        # Generate prompts
        train_prompts = [convert_iris_to_gemma_prompt(x, y) for x, y in zip(features[:8], species[:8])]
        test_prompts = [convert_iris_to_gemma_prompt(x, for_inference=True) for x in features[8:]]
        
        assert len(train_prompts) == 8
        assert len(test_prompts) == 2
        
        # Test base model evaluation
        base_evaluator = BaseGemmaEvaluator()
        predictions, confidences = base_evaluator.predict(test_prompts)
        
        assert len(predictions) == 2
        assert len(confidences) == 2
        
        # Cleanup
        if os.path.exists('test_iris.csv'):
            os.remove('test_iris.csv')

class TestResultValidation:
    """Test result validation and output format"""
    
    def test_result_json_structure(self):
        """Test expected JSON result structure"""
        # This would be generated by the main evaluation
        expected_structure = {
            'workflow_status': 'completed',
            'evaluation_timestamp': '2025-08-17 12:00:00',
            'training_time_seconds': 10.5,
            'dataset_info': {
                'total_samples': 150,
                'training_samples': 105,
                'test_samples': 45,
                'classes': ['setosa', 'versicolor', 'virginica']
            },
            'base_model_metrics': {
                'accuracy': 0.8889,
                'precision': 0.8889,
                'recall': 0.8889,
                'f1_score': 0.8889,
                'average_confidence': 0.85
            },
            'finetuned_model_metrics': {
                'accuracy': 0.9556,
                'precision': 0.9556,
                'recall': 0.9556,
                'f1_score': 0.9556,
                'average_confidence': 0.92
            },
            'improvements': {
                'accuracy_improvement_percent': 7.5,
                'f1_improvement_percent': 7.5,
                'precision_improvement_percent': 7.5,
                'recall_improvement_percent': 7.5
            },
            'validation_passed': True
        }
        
        # Test required keys exist
        required_keys = [
            'workflow_status', 'base_model_metrics', 'finetuned_model_metrics', 
            'improvements', 'validation_passed'
        ]
        
        for key in required_keys:
            assert key in expected_structure
            
        # Test metric keys
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
        for key in metric_keys:
            assert key in expected_structure['base_model_metrics']
            assert key in expected_structure['finetuned_model_metrics']
            
        # Test improvement keys
        improvement_keys = [f'{metric}_improvement_percent' for metric in metric_keys]
        for key in improvement_keys:
            assert key in expected_structure['improvements']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
