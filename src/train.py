import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_model():
    df = pd.read_csv('data/iris.csv')
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
    
    model = LogisticRegression(random_state=42, max_iter=1000)
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