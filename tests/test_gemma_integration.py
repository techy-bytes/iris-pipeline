"""
Test suite for Gemma LLM integration with IRIS classification.
Tests both conversion to linguistic format and model comparison.
"""

import pytest
import pandas as pd
import os
import json
from scripts.convert_iris_to_linguistic import convert_iris_to_linguistic
from scripts.compare_models import compare_models

def test_iris_conversion_to_linguistic():
    """Test conversion of IRIS data to linguistic format."""
    # Ensure clean state
    for file in ['data/train.jsonl', 'data/eval.jsonl', 'data/iris_sample.json']:
        if os.path.exists(file):
            os.remove(file)
    
    # Convert data
    train_file, eval_file, sample_file = convert_iris_to_linguistic()
    
    # Verify files were created
    assert os.path.exists(train_file), "Training file should be created"
    assert os.path.exists(eval_file), "Evaluation file should be created"
    assert os.path.exists(sample_file), "Sample file should be created"
    
    # Check training data format
    with open(train_file, 'r') as f:
        first_line = f.readline()
        data = json.loads(first_line)
    
    # Validate structure
    assert "messages" in data, "Should have messages key"
    assert len(data["messages"]) == 3, "Should have system, user, assistant messages"
    assert data["messages"][0]["role"] == "system", "First message should be system"
    assert data["messages"][1]["role"] == "user", "Second message should be user"
    assert data["messages"][2]["role"] == "assistant", "Third message should be assistant"
    
    # Validate content
    system_content = data["messages"][0]["content"]
    assert "Classify the flower" in system_content, "System should contain classification instruction"
    assert "[Setosa, Versicolor, Virginica]" in system_content, "System should list species"
    
    user_content = data["messages"][1]["content"]
    assert "Sepal Length:" in user_content, "User message should contain sepal length"
    assert "Petal Length:" in user_content, "User message should contain petal length"
    
    assistant_content = data["messages"][2]["content"]
    assert assistant_content in ["Setosa", "Versicolor", "Virginica"], "Assistant should predict valid species"
    
    # Check sample file structure
    with open(sample_file, 'r') as f:
        sample_data = json.load(f)
    
    assert "setosa_sample" in sample_data, "Should have setosa sample"
    assert "versicolor_sample" in sample_data, "Should have versicolor sample"
    assert "virginica_sample" in sample_data, "Should have virginica sample"
    
    # Verify each sample has different species
    species_found = set()
    for key in ["setosa_sample", "versicolor_sample", "virginica_sample"]:
        if sample_data[key]:  # Check if not None
            species = sample_data[key]["messages"][2]["content"]
            species_found.add(species)
    
    assert len(species_found) >= 2, f"Should have samples from different species, found: {species_found}"
    
    print("✅ IRIS data conversion to linguistic format successful")


def test_data_split_proportions():
    """Test that data is split correctly between train and eval."""
    train_file, eval_file, _ = convert_iris_to_linguistic()
    
    # Count lines in each file
    with open(train_file, 'r') as f:
        train_count = sum(1 for line in f)
    
    with open(eval_file, 'r') as f:
        eval_count = sum(1 for line in f)
    
    total_samples = train_count + eval_count
    train_ratio = train_count / total_samples
    eval_ratio = eval_count / total_samples
    
    # Should be approximately 80/20 split
    assert 0.75 <= train_ratio <= 0.85, f"Training ratio should be ~0.8, got {train_ratio:.3f}"
    assert 0.15 <= eval_ratio <= 0.25, f"Evaluation ratio should be ~0.2, got {eval_ratio:.3f}"
    assert total_samples == 150, f"Total samples should be 150, got {total_samples}"
    
    print(f"✅ Data split verified: {train_count} train, {eval_count} eval samples")


def test_model_comparison():
    """Test model comparison between LogisticRegression and Gemma LLM."""
    # Clean up previous results
    comparison_files = [
        'model_comparison_lr_vs_gemma.csv',
        'model_comparison_results.json'
    ]
    for file in comparison_files:
        if os.path.exists(file):
            os.remove(file)
    
    # Run comparison
    results = compare_models()
    
    # Verify results structure
    assert "logistic_regression" in results, "Should have LogisticRegression results"
    assert "gemma_llm" in results, "Should have Gemma LLM results"
    assert "efficiency_comparison" in results, "Should have efficiency comparison"
    assert "recommendation" in results, "Should have recommendation"
    
    # Verify LogisticRegression metrics
    lr_metrics = results["logistic_regression"]
    assert lr_metrics["accuracy"] > 0.8, f"LR accuracy should be > 0.8, got {lr_metrics['accuracy']}"
    assert lr_metrics["training_time"] < 1.0, f"LR training should be < 1s, got {lr_metrics['training_time']}"
    assert lr_metrics["num_parameters"] < 100, f"LR should have < 100 parameters, got {lr_metrics['num_parameters']}"
    
    # Verify Gemma metrics (simulated)
    gemma_metrics = results["gemma_llm"]
    assert gemma_metrics["accuracy"] > 0.8, f"Gemma accuracy should be > 0.8, got {gemma_metrics['accuracy']}"
    assert gemma_metrics["num_parameters"] > 1000000, f"Gemma should have > 1M parameters, got {gemma_metrics['num_parameters']}"
    
    # Verify efficiency comparison
    efficiency = results["efficiency_comparison"]
    assert efficiency["lr_vs_gemma_training_speedup"] > 1000, "LR should be much faster to train"
    assert efficiency["lr_vs_gemma_parameter_ratio"] < 0.001, "LR should have much fewer parameters"
    
    # Verify output files
    assert os.path.exists('model_comparison_lr_vs_gemma.csv'), "CSV comparison file should be created"
    assert os.path.exists('model_comparison_results.json'), "JSON results file should be created"
    
    # Verify CSV structure
    comparison_df = pd.read_csv('model_comparison_lr_vs_gemma.csv')
    expected_metrics = [
        'lr_accuracy', 'gemma_accuracy',
        'lr_training_time', 'gemma_training_time',
        'lr_prediction_time', 'gemma_prediction_time',
        'lr_parameters', 'gemma_parameters'
    ]
    for metric in expected_metrics:
        assert metric in comparison_df['metric'].values, f"Metric {metric} should be in CSV"
    
    print("✅ Model comparison test successful")
    print(f"   LR Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"   Gemma Accuracy: {gemma_metrics['accuracy']:.4f}")
    print(f"   Training speedup (LR vs Gemma): {efficiency['lr_vs_gemma_training_speedup']:.1f}x")


def test_linguistic_format_quality():
    """Test quality of linguistic format conversion."""
    # Load sample data
    sample_file = 'data/iris_sample.json'
    if not os.path.exists(sample_file):
        convert_iris_to_linguistic()
    
    with open(sample_file, 'r') as f:
        sample_data = json.load(f)
    
    # Test each sample
    for sample_key, sample in sample_data.items():
        if sample is None:
            continue
            
        user_message = sample["messages"][1]["content"]
        assistant_message = sample["messages"][2]["content"]
        
        # Verify user message contains all measurements
        measurements = ["Sepal Length:", "Sepal Width:", "Petal Length:", "Petal Width:"]
        for measurement in measurements:
            assert measurement in user_message, f"User message should contain {measurement}"
        
        # Verify assistant response is valid species
        assert assistant_message in ["Setosa", "Versicolor", "Virginica"], \
            f"Assistant response should be valid species, got: {assistant_message}"
        
        # Verify numerical values are present
        import re
        numbers = re.findall(r'\d+\.\d+', user_message)
        assert len(numbers) == 4, f"Should have 4 numerical measurements, found {len(numbers)}"
    
    print("✅ Linguistic format quality verified")


def test_llm_integration_readiness():
    """Test that the pipeline is ready for LLM integration."""
    # Check required files exist
    required_files = [
        'data/train.jsonl',
        'data/eval.jsonl', 
        'data/iris_sample.json',
        'scripts/convert_iris_to_linguistic.py',
        'scripts/compare_models.py'
    ]
    
    for file in required_files:
        assert os.path.exists(file), f"Required file missing: {file}"
    
    # Check data format consistency
    with open('data/train.jsonl', 'r') as f:
        first_sample = json.loads(f.readline())
        
    # Verify the format matches Gemma's expected input
    assert len(first_sample["messages"]) == 3, "Should have 3 messages for chat format"
    assert first_sample["messages"][0]["role"] == "system", "First should be system message"
    
    # Check that requirements include LLM dependencies
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
        
    llm_deps = ['transformers', 'torch', 'peft', 'google-cloud-storage']
    for dep in llm_deps:
        assert dep in requirements, f"Missing LLM dependency: {dep}"
    
    print("✅ LLM integration readiness verified")
    print("   - Linguistic data format ready")
    print("   - Comparison framework ready")
    print("   - Dependencies specified")


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield
    # Keep important files for CI/CD pipeline
    # Only clean up temporary test files
    pass