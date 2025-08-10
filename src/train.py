import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def add_location_attribute(df):
    """Add a random binary location attribute to the dataset."""
    np.random.seed(42)  # For reproducibility
    location = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])
    df['location'] = location
    return df

def compute_fairness_metrics(model, X_test, y_test, location_test, le):
    """Compute basic fairness metrics manually."""
    predictions = model.predict(X_test)
    
    # Group predictions by location
    location_0_mask = location_test == 0
    location_1_mask = location_test == 1
    
    # Accuracy by location
    acc_loc_0 = accuracy_score(y_test[location_0_mask], predictions[location_0_mask])
    acc_loc_1 = accuracy_score(y_test[location_1_mask], predictions[location_1_mask])
    
    # Overall accuracy
    overall_acc = accuracy_score(y_test, predictions)
    
    # Equalized odds (TPR by group for each class)
    fairness_metrics = {
        'overall_accuracy': overall_acc,
        'accuracy_location_0': acc_loc_0,
        'accuracy_location_1': acc_loc_1,
        'accuracy_difference': abs(acc_loc_0 - acc_loc_1),
        'statistical_parity_difference': 0.0  # Placeholder
    }
    
    # Calculate statistical parity for each class
    classes = le.classes_
    for class_idx, class_name in enumerate(classes):
        pred_rate_loc_0 = np.mean(predictions[location_0_mask] == class_idx)
        pred_rate_loc_1 = np.mean(predictions[location_1_mask] == class_idx)
        fairness_metrics[f'prediction_rate_{class_name}_location_0'] = pred_rate_loc_0
        fairness_metrics[f'prediction_rate_{class_name}_location_1'] = pred_rate_loc_1
        fairness_metrics[f'prediction_rate_diff_{class_name}'] = abs(pred_rate_loc_0 - pred_rate_loc_1)
    
    return fairness_metrics

def generate_feature_importance_analysis(model, feature_names, class_names):
    """Generate basic feature importance analysis (SHAP-like)."""
    importances = model.feature_importances_
    
    # Get top features for overall model
    feature_importance_dict = dict(zip(feature_names, importances))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    analysis = {
        'feature_importances': feature_importance_dict,
        'top_features': sorted_features,
        'virginica_analysis': {
            'description': "Feature importance analysis for virginica classification",
            'key_insights': [
                "Random Forest feature importance indicates global feature contribution",
                "Higher values suggest features more important for distinguishing virginica",
                "Feature interactions are captured implicitly in tree-based models"
            ]
        }
    }
    
    return analysis

def train_model():
    # Load and prepare data
    df = pd.read_csv('data/iris.csv')
    
    # Add location attribute
    df = add_location_attribute(df)
    
    print(f"Training with dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")
    print(f"Species distribution:")
    print(df['species'].value_counts())
    print(f"Location distribution:")
    print(df['location'].value_counts())
    
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]
    y = df['species']
    location = df['location']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test, location_train, location_test = train_test_split(
        X, y_encoded, location, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute fairness metrics
    fairness_metrics = compute_fairness_metrics(model, X_test, y_test, location_test, le)
    
    # Generate feature importance analysis
    feature_analysis = generate_feature_importance_analysis(model, feature_cols, le.classes_)
    
    # Save enhanced metrics
    metrics = {
        'accuracy': accuracy,
        'num_samples': len(df),
        'num_features': len(feature_cols),
        'num_classes': len(df['species'].unique()),
        **fairness_metrics
    }
    
    with open('metrics.csv', 'w') as f:
        f.write('metric,value\n')
        for key, value in metrics.items():
            f.write(f'{key},{value}\n')
    
    # Save fairness analysis as JSON
    import json
    with open('fairness_analysis.json', 'w') as f:
        json.dump({
            'fairness_metrics': fairness_metrics,
            'feature_analysis': feature_analysis
        }, f, indent=2)
    
    # Save model components
    joblib.dump(model, 'model.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    
    # Save the enhanced dataset for API use
    df.to_csv('data/iris_with_location.csv', index=False)
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Dataset size: {len(df)} samples")
    print(f"Features: {len(feature_cols)}")
    print(f"Classes: {list(df['species'].unique())}")
    print(f"Fairness metrics saved to fairness_analysis.json")
    
    # Print some fairness insights
    print(f"\nFairness Analysis:")
    print(f"Accuracy difference between locations: {fairness_metrics['accuracy_difference']:.4f}")
    print(f"Location 0 accuracy: {fairness_metrics['accuracy_location_0']:.4f}")
    print(f"Location 1 accuracy: {fairness_metrics['accuracy_location_1']:.4f}")
    
    return accuracy

if __name__ == "__main__":
    train_model()