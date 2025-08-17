import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Dict, Any, Optional
from src.gemma_classifier import MockGemmaIrisClassifier, LocalGemmaIrisClassifier


def train_sklearn_model(csv_path: str = 'data/iris.csv') -> float:
    """Train the original sklearn RandomForest model."""
    df = pd.read_csv(csv_path)
    print(f"Training sklearn model with dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")
    print(f"Species distribution:")
    print(df['species'].value_counts())
    
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]
    y = df['species']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model artifacts
    joblib.dump(model, 'model.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    
    print(f"Sklearn model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy


def train_gemma_model_mock() -> float:
    """Train/evaluate the mock Gemma model."""
    print("Training/evaluating mock Gemma model...")
    
    # Use mock classifier for testing
    classifier = MockGemmaIrisClassifier()
    classifier.load_model()
    
    # Evaluate on test data
    metrics = classifier.evaluate_on_test_data('data/iris.csv')
    accuracy = metrics['accuracy']
    
    print(f"Mock Gemma model evaluation complete!")
    print(f"Accuracy: {accuracy:.4f}")
    
    classifier.cleanup()
    return accuracy


def train_gemma_model_local(local_model_path: str = "models/iris-gemma-local", 
                           hf_token: Optional[str] = None) -> float:
    """Train/evaluate the local Gemma model."""
    print(f"Evaluating local Gemma model from {local_model_path}...")
    
    # Check if local model exists
    if not os.path.exists(local_model_path):
        print(f"❌ Local model not found at {local_model_path}")
        print("💡 Train the model first using kaggle_train.py or the Kaggle notebook")
        return 0.0
    
    try:
        # Use local classifier
        classifier = LocalGemmaIrisClassifier(
            local_model_path=local_model_path,
            hf_token=hf_token
        )
        
        if not classifier.load_model():
            print("❌ Failed to load local model")
            return 0.0
        
        # Evaluate on test data
        metrics = classifier.evaluate_on_test_data('data/iris.csv')
        accuracy = metrics['accuracy']
        
        print(f"Local Gemma model evaluation complete!")
        print(f"Accuracy: {accuracy:.4f}")
        
        classifier.cleanup()
        return accuracy
        
    except Exception as e:
        print(f"❌ Error evaluating local model: {e}")
        return 0.0


def compare_models(use_local_gemma: bool = False, 
                  local_model_path: str = "models/iris-gemma-local",
                  hf_token: Optional[str] = None) -> Dict[str, Any]:
    """Train and compare both models."""
    print("=" * 50)
    print("COMPARING IRIS CLASSIFICATION MODELS")
    print("=" * 50)
    
    # Train sklearn model
    print("\n1. Training Sklearn RandomForest Model:")
    sklearn_accuracy = train_sklearn_model()
    
    # Train/evaluate Gemma model
    if use_local_gemma:
        print("\n2. Evaluating Local Gemma LLM Model:")
        gemma_accuracy = train_gemma_model_local(local_model_path, hf_token)
        model_type = "local_gemma"
    else:
        print("\n2. Evaluating Mock Gemma LLM Model:")
        gemma_accuracy = train_gemma_model_mock()
        model_type = "mock_gemma"
    
    # Compare results
    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS")
    print("=" * 50)
    print(f"Sklearn RandomForest Accuracy: {sklearn_accuracy:.4f}")
    print(f"Gemma LLM Accuracy:           {gemma_accuracy:.4f}")
    
    if gemma_accuracy > sklearn_accuracy:
        print("🏆 Gemma LLM model performs better!")
        winner = "gemma"
    elif sklearn_accuracy > gemma_accuracy:
        print("🏆 Sklearn RandomForest model performs better!")
        winner = "sklearn"
    else:
        print("🤝 Both models perform equally well!")
        winner = "tie"
    
    comparison_results = {
        'sklearn_accuracy': sklearn_accuracy,
        'gemma_accuracy': gemma_accuracy,
        'winner': winner,
        'accuracy_difference': abs(sklearn_accuracy - gemma_accuracy),
        'gemma_model_type': model_type
    }
    
    return comparison_results


def train_model(model_type: str = 'sklearn') -> float:
    """Train the specified model type. Maintains backward compatibility."""
    if model_type == 'sklearn':
        return train_sklearn_model()
    elif model_type == 'gemma':
        return train_gemma_model_mock()
    elif model_type == 'compare':
        results = compare_models()
        return max(results['sklearn_accuracy'], results['gemma_accuracy'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_metrics_file(sklearn_accuracy: Optional[float] = None, 
                       gemma_accuracy: Optional[float] = None):
    """Create metrics.csv file with model comparison results."""
    
    # If no accuracies provided, train/evaluate both models
    if sklearn_accuracy is None:
        sklearn_accuracy = train_sklearn_model()
    if gemma_accuracy is None:
        gemma_accuracy = train_gemma_model_mock()
    
    # Load dataset info for metrics
    df = pd.read_csv('data/iris.csv')
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Use the better performing model's accuracy as primary metric
    primary_accuracy = max(sklearn_accuracy, gemma_accuracy)
    best_model = "gemma_llm" if gemma_accuracy > sklearn_accuracy else "sklearn_rf"
    
    metrics = {
        'accuracy': primary_accuracy,
        'sklearn_accuracy': sklearn_accuracy,
        'gemma_accuracy': gemma_accuracy,
        'best_model': best_model,
        'accuracy_improvement': abs(sklearn_accuracy - gemma_accuracy),
        'num_samples': len(df),
        'num_features': len(feature_cols),
        'num_classes': len(df['species'].unique())
    }
    
    with open('metrics.csv', 'w') as f:
        f.write('metric,value\n')
        for key, value in metrics.items():
            if isinstance(value, str):
                f.write(f'{key},"{value}"\n')
            else:
                f.write(f'{key},{value}\n')
    
    print(f"\n✅ Metrics saved to metrics.csv")
    print(f"Primary accuracy: {primary_accuracy:.4f} (using {best_model})")
    
    return primary_accuracy


if __name__ == "__main__":
    # Default behavior: compare both models and create metrics
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate Iris classification models")
    parser.add_argument("--model_type", default="compare", choices=["sklearn", "gemma", "compare", "local"],
                       help="Model type to train/evaluate")
    parser.add_argument("--local_model_path", default="models/iris-gemma-local",
                       help="Path to local Gemma model")
    parser.add_argument("--hf_token", default=None,
                       help="HuggingFace token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Get HF token from args or environment
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    
    if args.model_type == "local":
        # Use local Gemma model
        results = compare_models(
            use_local_gemma=True, 
            local_model_path=args.local_model_path,
            hf_token=hf_token
        )
        create_metrics_file(results['sklearn_accuracy'], results['gemma_accuracy'])
    elif args.model_type == "compare":
        # Default: compare mock models and create comprehensive metrics
        results = compare_models()
        create_metrics_file(results['sklearn_accuracy'], results['gemma_accuracy'])
    else:
        # Single model training
        accuracy = train_model(args.model_type)
        create_metrics_file()