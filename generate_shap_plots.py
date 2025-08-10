#!/usr/bin/env python3
"""
Generate SHAP explainer plots for the IRIS dataset.
This script creates comprehensive SHAP visualizations to understand model predictions.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

# Set up matplotlib for non-interactive backend
plt.switch_backend('Agg')

def load_model_and_data():
    """Load the trained model and prepared dataset."""
    # Load the trained model and label encoder
    model = joblib.load('model.joblib')
    le = joblib.load('label_encoder.joblib')
    
    # Load the dataset with location attribute
    df = pd.read_csv('data/iris_with_location.csv')
    
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]
    y = df['species']
    location = df['location']
    
    y_encoded = le.transform(y)
    
    # Use the same split as in training
    X_train, X_test, y_train, y_test, location_train, location_test = train_test_split(
        X, y_encoded, location, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    return model, le, X_train, X_test, y_train, y_test, location_train, location_test, feature_cols

def generate_shap_plots():
    """Generate comprehensive SHAP explainer plots."""
    # Load model and data
    model, le, X_train, X_test, y_train, y_test, location_train, location_test, feature_cols = load_model_and_data()
    
    print("Generating SHAP explainer plots...")
    print(f"Test set shape: {X_test.shape}")
    print(f"Feature columns: {feature_cols}")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)
    
    print(f"SHAP values type: {type(shap_values)}")
    print(f"SHAP values shape: {shap_values.shape}")
    
    # SHAP values have shape (samples, features, classes)
    # For multiclass, we need to transpose to get (classes, samples, features)
    if len(shap_values.shape) == 3:
        # Transpose from (samples, features, classes) to (classes, samples, features)
        shap_values_transposed = shap_values.transpose(2, 0, 1)
        virginica_shap_values = shap_values_transposed[2]  # Class 2 is virginica
    else:
        # For binary classification or other formats
        virginica_shap_values = shap_values
    
    print(f"Virginica SHAP values shape: {virginica_shap_values.shape}")
    print(f"Expected shape: ({X_test.shape[0]}, {X_test.shape[1]})")
    
    # Create plots directory
    os.makedirs('shap_plots', exist_ok=True)
    
    # 1. Summary Plot - shows feature importance and impact for all samples
    plt.figure(figsize=(10, 6))
    shap.summary_plot(virginica_shap_values, X_test, feature_names=feature_cols, 
                     title="SHAP Summary Plot - Virginica Classification", show=False)
    plt.tight_layout()
    plt.savefig('shap_plots/shap_summary_virginica.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Summary Plot (bar) - shows feature importance ranking
    plt.figure(figsize=(10, 6))
    shap.summary_plot(virginica_shap_values, X_test, feature_names=feature_cols, 
                     plot_type="bar", title="Feature Importance - Virginica Classification", show=False)
    plt.tight_layout()
    plt.savefig('shap_plots/shap_feature_importance_virginica.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Waterfall plot for a specific prediction (highest virginica prediction)
    # Find the sample with highest virginica prediction
    virginica_probs = model.predict_proba(X_test)[:, 2]
    max_virginica_idx = np.argmax(virginica_probs)
    
    # Create a simple explanation object for waterfall plot
    plt.figure(figsize=(10, 6))
    shap_explanation = shap.Explanation(
        values=virginica_shap_values[max_virginica_idx],
        base_values=explainer.expected_value[2],
        data=X_test.iloc[max_virginica_idx].values,
        feature_names=feature_cols
    )
    shap.waterfall_plot(shap_explanation, show=False)
    plt.title(f"SHAP Waterfall Plot - Sample with Highest Virginica Prediction\n"
              f"Prediction Probability: {virginica_probs[max_virginica_idx]:.3f}")
    plt.tight_layout()
    plt.savefig('shap_plots/shap_waterfall_best_virginica.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature dependence plots for top 2 features
    top_features = ['petal_width', 'petal_length']  # Based on our analysis
    
    for i, feature in enumerate(top_features):
        feature_idx = feature_cols.index(feature)
        plt.figure(figsize=(10, 6))
        
        # Create a simple dependence plot manually since partial_dependence_plot may have issues
        feature_values = X_test[feature].values
        shap_values_for_feature = virginica_shap_values[:, feature_idx]
        
        plt.scatter(feature_values, shap_values_for_feature, alpha=0.6, s=30)
        plt.xlabel(f"{feature.replace('_', ' ').title()}")
        plt.ylabel(f"SHAP value for {feature.replace('_', ' ')}")
        plt.title(f"SHAP Dependence Plot - {feature.replace('_', ' ').title()}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'shap_plots/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Decision plot showing the path from base value to prediction
    plt.figure(figsize=(10, 8))
    shap.decision_plot(explainer.expected_value[2], virginica_shap_values[:20], 
                      X_test.iloc[:20], feature_names=feature_cols, show=False)
    plt.title("SHAP Decision Plot - First 20 Test Samples\nPath from Base Value to Prediction")
    plt.tight_layout()
    plt.savefig('shap_plots/shap_decision_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("SHAP plots generated successfully!")
    print("Files created in 'shap_plots/' directory:")
    for file in sorted(os.listdir('shap_plots')):
        print(f"  - {file}")
    
    # Generate summary statistics
    feature_importance = np.abs(virginica_shap_values).mean(0)
    feature_importance_pct = 100 * feature_importance / feature_importance.sum()
    
    print("\nFeature Importance Summary (from SHAP values):")
    for i, (feature, importance) in enumerate(zip(feature_cols, feature_importance_pct)):
        print(f"{i+1}. {feature}: {importance:.1f}%")
    
    # Compare by location
    location_0_mask = location_test == 0
    location_1_mask = location_test == 1
    
    print(f"\nSample Distribution in Test Set:")
    print(f"Location 0: {location_0_mask.sum()} samples")
    print(f"Location 1: {location_1_mask.sum()} samples")
    
    return {
        'feature_importance_pct': dict(zip(feature_cols, feature_importance_pct)),
        'num_plots_generated': len(os.listdir('shap_plots')),
        'virginica_prediction_stats': {
            'min_prob': float(virginica_probs.min()),
            'max_prob': float(virginica_probs.max()),
            'mean_prob': float(virginica_probs.mean())
        }
    }

if __name__ == "__main__":
    try:
        results = generate_shap_plots()
        print(f"\n✅ Successfully generated {results['num_plots_generated']} SHAP plots!")
    except Exception as e:
        print(f"❌ Error generating SHAP plots: {e}")
        import traceback
        traceback.print_exc()