import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Dict, Any, Optional
from src.gemma_classifier import MockGemmaIrisClassifier


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


def compare_models() -> Dict[str, Any]:
    """Train and compare both models."""
    print("=" * 50)
    print("COMPARING IRIS CLASSIFICATION MODELS")
    print("=" * 50)
    
    # Train sklearn model
    print("\n1. Training Sklearn RandomForest Model:")
    sklearn_accuracy = train_sklearn_model()
    
    # Train/evaluate Gemma model
    print("\n2. Evaluating Gemma LLM Model:")
    gemma_accuracy = train_gemma_model_mock()
    
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
        'accuracy_difference': abs(sklearn_accuracy - gemma_accuracy)
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
    
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        accuracy = train_model(model_type)
        create_metrics_file()
    else:
        # Default: compare models and create comprehensive metrics
        results = compare_models()
        create_metrics_file(results['sklearn_accuracy'], results['gemma_accuracy'])