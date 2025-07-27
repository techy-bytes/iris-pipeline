from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from typing import List
from contextlib import asynccontextmanager

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add console exporter for demonstration (in production, you'd use a proper exporter)
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Global variables for model and label encoder
model = None
label_encoder = None
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    species: str
    confidence: float

def load_model():
    """Load the trained model and label encoder."""
    global model, label_encoder
    
    if os.path.exists('model.joblib') and os.path.exists('label_encoder.joblib'):
        model = joblib.load('model.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        return True
    return False

def train_and_save_model():
    """Train the model and save it for inference."""
    global model, label_encoder
    
    # Try different possible data paths
    data_paths = ['data/iris.csv', '../data/iris.csv', './data/iris.csv']
    df = None
    
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError("Could not find iris.csv in any of the expected locations")
    
    # Prepare features and target
    X = df[feature_names]
    y = df['species']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train model
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y_encoded)
    
    # Save model and encoder
    joblib.dump(model, 'model.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')
    
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model on startup."""
    global model, label_encoder
    if not load_model():
        print("Model not found. Training new model...")
        train_and_save_model()
        print("Model trained and saved successfully.")
    else:
        print("Model loaded successfully.")
    yield

app = FastAPI(title="Iris Classifier API", version="1.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Iris Classifier API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_names": feature_names,
        "classes": label_encoder.classes_.tolist()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """Predict iris species based on features."""
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features for prediction
        feature_array = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]
        
        # Decode prediction
        species = label_encoder.inverse_transform([prediction])[0]
        confidence = float(max(probabilities))
        
        return PredictionResponse(species=species, confidence=confidence)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(features_list: List[IrisFeatures]):
    """Predict iris species for multiple samples."""
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not features_list:
        return []
    
    try:
        # Prepare features for prediction
        feature_arrays = []
        for features in features_list:
            feature_arrays.append([
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ])
        
        feature_matrix = np.array(feature_arrays)
        
        # Make predictions with OpenTelemetry instrumentation
        with tracer.start_as_current_span("model_prediction_batch") as span:
            # Add span attributes for observability
            span.set_attribute("batch_size", len(features_list))
            span.set_attribute("model_type", "RandomForestClassifier")
            span.set_attribute("feature_count", feature_matrix.shape[1])
            
            predictions = model.predict(feature_matrix)
            probabilities = model.predict_proba(feature_matrix)
        
        # Decode predictions
        results = []
        for i, pred in enumerate(predictions):
            species = label_encoder.inverse_transform([pred])[0]
            confidence = float(max(probabilities[i]))
            results.append(PredictionResponse(species=species, confidence=confidence))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)