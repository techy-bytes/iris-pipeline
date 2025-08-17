import pytest
import pandas as pd
import json
import os
from src.data_transformer import convert_iris_to_linguistic_format, get_linguistic_prediction_prompt


class TestDataTransformer:
    """Test data transformation utilities."""
    
    def test_convert_iris_to_linguistic_format(self, tmp_path):
        """Test conversion of iris data to linguistic format."""
        # Create test CSV
        test_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 6.0],
            'sepal_width': [3.5, 3.0, 3.0],
            'petal_length': [1.4, 1.4, 4.5],
            'petal_width': [0.2, 0.2, 1.5],
            'species': ['setosa', 'setosa', 'versicolor']
        })
        
        input_csv = tmp_path / "test_iris.csv"
        test_data.to_csv(input_csv, index=False)
        
        train_output = tmp_path / "train.jsonl"
        eval_output = tmp_path / "eval.jsonl"
        
        # Convert to linguistic format
        train_count, eval_count = convert_iris_to_linguistic_format(
            str(input_csv), str(train_output), str(eval_output), train_split=0.7
        )
        
        # Check outputs
        assert train_count > 0
        assert eval_count > 0
        assert train_count + eval_count == len(test_data)
        assert train_output.exists()
        assert eval_output.exists()
        
        # Check format of training data
        with open(train_output, 'r') as f:
            first_line = f.readline()
            example = json.loads(first_line)
            
            assert "messages" in example
            assert len(example["messages"]) == 3
            assert example["messages"][0]["role"] == "system"
            assert example["messages"][1]["role"] == "user"
            assert example["messages"][2]["role"] == "assistant"
            assert "Sepal Length:" in example["messages"][1]["content"]
            assert example["messages"][2]["content"] in ["Setosa", "Versicolor", "Virginica"]
    
    def test_get_linguistic_prediction_prompt(self):
        """Test generation of prediction prompts."""
        messages = get_linguistic_prediction_prompt(5.1, 3.5, 1.4, 0.2)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Sepal Length: 5.1" in messages[1]["content"]
        assert "Sepal Width: 3.5" in messages[1]["content"]
        assert "Petal Length: 1.4" in messages[1]["content"]
        assert "Petal Width: 0.2" in messages[1]["content"]
        assert "Classify the flower" in messages[0]["content"]