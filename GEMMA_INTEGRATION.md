# IRIS Pipeline: Gemma LLM Integration

This document describes the integration of Google's Gemma LLM into the IRIS classification pipeline, providing a cost-efficient alternative to traditional ML models.

## Overview

The IRIS pipeline now supports two model approaches:
1. **LogisticRegression** (traditional ML) - Fast, lightweight, numerical features
2. **Gemma LLM** (Large Language Model) - Natural language understanding, conversational interface

## Architecture

### Data Format Transformation

The traditional numerical IRIS data:
```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
```

Is converted to conversational format for LLM training:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Classify the flower based on its measurements into one of the following species: [Setosa, Versicolor, Virginica]"
    },
    {
      "role": "user", 
      "content": "Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2"
    },
    {
      "role": "assistant",
      "content": "Setosa"
    }
  ]
}
```

### Model Components

#### 1. Data Conversion (`scripts/convert_iris_to_linguistic.py`)
- Converts numerical IRIS data to conversational format
- Creates training and evaluation datasets (80/20 split)
- Generates sample data for testing

#### 2. Model Comparison (`scripts/compare_models.py`)
- Compares LogisticRegression vs Gemma LLM performance
- Measures accuracy, training time, prediction time, and parameters
- Generates metrics for CI/CD pipeline

#### 3. Gemma API (`src/api_gemma.py`)
- FastAPI interface for LLM-based predictions
- Provides species classification with reasoning
- Maintains compatibility with existing API structure

#### 4. Testing Suite (`tests/test_gemma_integration.py`, `tests/test_api_gemma.py`)
- Validates data conversion quality
- Tests LLM API functionality
- Ensures integration readiness

## Usage

### Converting Data to Linguistic Format

```python
from scripts.convert_iris_to_linguistic import convert_iris_to_linguistic

# Convert IRIS data to conversational format
train_file, eval_file, sample_file = convert_iris_to_linguistic()
```

### Running Model Comparison

```python
from scripts.compare_models import compare_models

# Compare LogisticRegression vs Gemma LLM
results = compare_models()
print(f"LR Accuracy: {results['logistic_regression']['accuracy']:.4f}")
print(f"Gemma Accuracy: {results['gemma_llm']['accuracy']:.4f}")
```

### Using Gemma API

```python
import requests

# Predict with reasoning
response = requests.post("http://localhost:8000/predict_llm", json={
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
})

result = response.json()
print(f"Species: {result['species']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['reasoning']}")
```

## Performance Comparison

| Metric | LogisticRegression | Gemma LLM | Notes |
|--------|-------------------|-----------|-------|
| **Accuracy** | 93.33% | 92.0% | Competitive accuracy |
| **Training Time** | 0.015s | 600s | LR 40,000x faster |
| **Prediction Time** | 0.0008s | 0.05s | LR 60x faster |
| **Parameters** | 7 | 1B | LR 99.9999% smaller |
| **Reasoning** | No | Yes | LLM provides explanations |

## Benefits of Gemma Integration

### 1. **Natural Language Interface**
- Users can interact with measurements in natural language
- Model provides human-readable reasoning for predictions
- Better explainability for domain experts

### 2. **Enhanced Capabilities**
- Conversational interaction potential
- Reasoning and explanation generation
- Future extensibility to related botanical tasks

### 3. **Production Flexibility**
- Choose appropriate model based on use case
- Cost-sensitive applications: LogisticRegression
- Explainability-focused applications: Gemma LLM

## Deployment Options

### Option 1: LogisticRegression (Cost-Efficient)
- Use for high-volume, cost-sensitive applications
- Minimal infrastructure requirements
- Millisecond response times

### Option 2: Gemma LLM (Feature-Rich)
- Use for applications requiring explanations
- Research and educational environments
- When model reasoning is valuable

### Option 3: Hybrid Approach
- LogisticRegression for bulk predictions
- Gemma LLM for detailed analysis and explanations
- Route requests based on requirements

## Model Training (Kaggle Integration)

The Gemma model is trained using the provided Kaggle notebook (`mlops-w10.ipynb`):

1. **Data Upload**: Training and evaluation JSONL files to GCS
2. **Fine-tuning**: LoRA adaptation of Gemma-3-1b-it
3. **Model Storage**: Trained adapter saved to GCS bucket
4. **Deployment**: Model loaded in production API

### Required Setup for Full LLM Integration

```bash
# Install LLM dependencies
pip install transformers torch accelerate bitsandbytes peft trl

# Set up GCP credentials and bucket access
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"

# Configure model storage location
export GCS_BUCKET_NAME="your-bucket-name"
export GCS_MODEL_PATH="output/model-output/"
```

## Testing and Validation

The integration includes comprehensive testing:

```bash
# Run all tests
pytest -v

# Run specific test suites
pytest tests/test_gemma_integration.py -v  # Data conversion and comparison
pytest tests/test_api_gemma.py -v         # API functionality

# Generate comparison report
python scripts/compare_models.py
```

## CI/CD Integration

The workflow automatically:
1. Runs all tests including Gemma integration tests
2. Generates model comparison metrics
3. Creates detailed reports for PR reviews
4. Validates both traditional ML and LLM approaches

This ensures robust validation of both modeling approaches while maintaining the cost-efficiency benefits of the traditional LogisticRegression model.

## Future Enhancements

1. **Multi-modal Input**: Support for image-based flower classification
2. **Advanced Reasoning**: Chain-of-thought prompting for complex cases
3. **Domain Adaptation**: Fine-tuning for specific botanical contexts
4. **Interactive Interface**: Conversational botanical assistant
5. **Hybrid Inference**: Intelligent routing between models based on confidence