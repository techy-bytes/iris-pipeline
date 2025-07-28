from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from typing import List
from contextlib import asynccontextmanager
import logging
import time
import json

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Initialize telemetry exporters based on environment
def setup_telemetry():
    """Setup telemetry exporters with fallback to console."""
    try:
        # Check if we have GCP credentials and project configured
        if os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCP_PROJECT'):
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
            gcp_exporter = CloudTraceSpanExporter()
            span_processor = BatchSpanProcessor(gcp_exporter)
            print("Using Google Cloud Trace exporter")
            return span_processor
    except Exception as e:
        print(f"Failed to initialize Google Cloud Trace exporter: {e}")
    
    # Fallback to console exporter for local development
    span_processor = BatchSpanProcessor(ConsoleSpanExporter())
    print("Using console exporter for traces")
    return span_processor

# Setup trace exporter
trace.get_tracer_provider().add_span_processor(setup_telemetry())

# Setup structured logging
import sys

def setup_logging():
    """Setup logging with Google Cloud Logging if available."""
    # Initialize Google Cloud Logging if available
    try:
        # Check if we're running in GCP environment with proper credentials
        if os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCP_PROJECT'):
            from google.cloud import logging as gcp_logging
            
            # Initialize Google Cloud Logging client
            gcp_client = gcp_logging.Client()
            gcp_client.setup_logging()
            print("Google Cloud Logging initialized")
            
            # Use Cloud Logging structured format
            logger = logging.getLogger("iris-ml-service")
            logger.setLevel(logging.INFO)
            return logger
        else:
            raise ImportError("GCP project not configured")
            
    except Exception as e:
        print(f"Using local logging setup: {e}")
        # Fallback to local structured logging
        logger = logging.getLogger("iris-ml-service")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        
        formatter = logging.Formatter(json.dumps({
            "severity": "%(levelname)s",
            "message": "%(message)s",
            "timestamp": "%(asctime)s"
        }))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

logger = setup_logging()

# Configure uvicorn and root logging to use stdout instead of stderr
# This ensures that all logs go to stdout, preventing Kubernetes from treating them as ERROR
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add a stdout handler to root logger
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
simple_formatter = logging.Formatter('%(levelname)s:     %(message)s')
stdout_handler.setFormatter(simple_formatter)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.INFO)

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

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "timestamp": time.time(),
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

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
async def predict(features: IrisFeatures, request: Request):
    """Predict iris species based on features."""
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")
        
        try:
            # Prepare features for prediction using DataFrame with feature names
            feature_data = pd.DataFrame({
                'sepal_length': [features.sepal_length],
                'sepal_width': [features.sepal_width],
                'petal_length': [features.petal_length],
                'petal_width': [features.petal_width]
            })
            
            # Make prediction
            prediction = model.predict(feature_data)[0]
            probabilities = model.predict_proba(feature_data)[0]
            
            # Decode prediction
            species = label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            latency = round((time.time() - start_time) * 1000, 2)
            
            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "timestamp": time.time(),
                "input": features.model_dump(),
                "result": {"species": species, "confidence": confidence},
                "latency_ms": latency,
                "status": "success"
            }))
            
            return PredictionResponse(species=species, confidence=confidence)
        
        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "timestamp": time.time(),
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(features_list: List[IrisFeatures]):
    """Predict iris species for multiple samples."""
    if model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not features_list:
        return []
    
    try:
        # Prepare features for prediction using DataFrame with feature names
        feature_data = []
        for features in features_list:
            feature_data.append({
                'sepal_length': features.sepal_length,
                'sepal_width': features.sepal_width,
                'petal_length': features.petal_length,
                'petal_width': features.petal_width
            })
        
        feature_df = pd.DataFrame(feature_data)
        
        # Make predictions with OpenTelemetry instrumentation
        with tracer.start_as_current_span("model_prediction_batch") as span:
            start_time = time.time()
            trace_id = format(span.get_span_context().trace_id, "032x")
            
            # Add span attributes for observability
            span.set_attribute("batch_size", len(features_list))
            span.set_attribute("model_type", "RandomForestClassifier")
            span.set_attribute("feature_count", feature_df.shape[1])
            
            predictions = model.predict(feature_df)
            probabilities = model.predict_proba(feature_df)
            
            latency = round((time.time() - start_time) * 1000, 2)
        
        # Decode predictions
        results = []
        for i, pred in enumerate(predictions):
            species = label_encoder.inverse_transform([pred])[0]
            confidence = float(max(probabilities[i]))
            results.append(PredictionResponse(species=species, confidence=confidence))
        
        logger.info(json.dumps({
            "event": "batch_prediction",
            "trace_id": trace_id,
            "timestamp": time.time(),
            "batch_size": len(features_list),
            "latency_ms": latency,
            "status": "success"
        }))
        
        return results
    
    except Exception as e:
        span = trace.get_current_span()
        trace_id = format(span.get_span_context().trace_id, "032x")
        logger.exception(json.dumps({
            "event": "batch_prediction_error",
            "trace_id": trace_id,
            "timestamp": time.time(),
            "error": str(e)
        }))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )