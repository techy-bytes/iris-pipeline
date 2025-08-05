import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_model():
    # Check if poisoned dataset exists and use it, otherwise use original
    poisoned_path = 'data/iris_poisoned.csv'
    original_path = 'data/iris.csv'
    
    if os.path.exists(poisoned_path):
        dataset_path = poisoned_path
        print(f"Using poisoned dataset: {dataset_path}")
    else:
        dataset_path = original_path
        print(f"Using original dataset: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"Training with dataset shape: {df.shape}")
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
    
    metrics = {
        'accuracy': accuracy,
        'num_samples': len(df),
        'num_features': len(feature_cols),
        'num_classes': len(df['species'].unique())
    }
    
    with open('metrics.csv', 'w') as f:
        f.write('metric,value\n')
        for key, value in metrics.items():
            f.write(f'{key},{value}\n')
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Dataset size: {len(df)} samples")
    print(f"Features: {len(feature_cols)}")
    print(f"Classes: {list(df['species'].unique())}")
    
    return accuracy

if __name__ == "__main__":
    train_model()