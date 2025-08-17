from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
from typing import List, Optional
from contextlib import asynccontextmanager
import logging
import time
import json

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

# Import our LLM classifier with error handling
try:
    from src.gemma_classifier import MockGemmaIrisClassifier, GemmaIrisClassifier
    LLM_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"LLM classifier not available: {e}")
    LLM_CLASSIFIER_AVAILABLE = False
    
    # Create dummy classes for fallback
    class MockGemmaIrisClassifier:
        def __init__(self, **kwargs):
            pass
        def load_model(self):
            return False
        def predict(self, *args):
            return "unknown", 0.0
    
    class GemmaIrisClassifier:
        def __init__(self, **kwargs):
            pass
        def load_model(self):
            return False

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add console exporter for demonstration
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
import sys

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

# Configure uvicorn and root logging to use stdout
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
simple_formatter = logging.Formatter('%(levelname)s:     %(message)s')
stdout_handler.setFormatter(simple_formatter)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.INFO)

# Global variables for models
sklearn_model = None
label_encoder = None
gemma_model = None
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Configuration
USE_LLM_MODEL = os.getenv('USE_LLM_MODEL', 'false').lower() == 'true'
USE_MOCK_LLM = os.getenv('USE_MOCK_LLM', 'true').lower() == 'true'

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    species: str
    confidence: float
    model_used: str

class ModelInfo(BaseModel):
    sklearn_available: bool
    gemma_available: bool
    default_model: str
    models_loaded: List[str]

def load_sklearn_model():
    """Load the trained sklearn model and label encoder."""
    global sklearn_model, label_encoder
    
    if os.path.exists('model.joblib') and os.path.exists('label_encoder.joblib'):
        sklearn_model = joblib.load('model.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        return True
    return False

def train_and_save_sklearn_model():
    """Train the sklearn model and save it for inference."""
    global sklearn_model, label_encoder
    
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
    sklearn_model = RandomForestClassifier(random_state=42, n_estimators=100)
    sklearn_model.fit(X, y_encoded)
    
    # Save model and encoder
    joblib.dump(sklearn_model, 'model.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')
    
    return True

def load_gemma_model():
    """Load the Gemma LLM model."""
    global gemma_model
    
    try:
        if USE_MOCK_LLM:
            print("Loading mock Gemma model...")
            gemma_model = MockGemmaIrisClassifier()
        else:
            print("Loading real Gemma model...")
            gemma_model = GemmaIrisClassifier()
        
        return gemma_model.load_model()
    except Exception as e:
        print(f"Failed to load Gemma model: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    global sklearn_model, label_encoder, gemma_model
    
    print("=" * 50)
    print("INITIALIZING IRIS CLASSIFICATION MODELS")
    print("=" * 50)
    
    # Load sklearn model
    if not load_sklearn_model():
        print("Sklearn model not found. Training new sklearn model...")
        train_and_save_sklearn_model()
        print("Sklearn model trained and saved successfully.")
    else:
        print("Sklearn model loaded successfully.")
    
    # Load Gemma model if configured
    if USE_LLM_MODEL:
        print("Loading Gemma LLM model...")
        if load_gemma_model():
            print("Gemma model loaded successfully.")
        else:
            print("Failed to load Gemma model. Will use sklearn only.")
    else:
        print("LLM model disabled (USE_LLM_MODEL=false)")
    
    print("=" * 50)
    yield

app = FastAPI(title="Enhanced Iris Classifier API", version="2.0.0", lifespan=lifespan)

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
    return {"message": "Enhanced Iris Classifier API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    models_loaded = []
    if sklearn_model is not None and label_encoder is not None:
        models_loaded.append("sklearn")
    if gemma_model is not None:
        models_loaded.append("gemma")
    
    if not models_loaded:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    default_model = "gemma" if USE_LLM_MODEL and gemma_model else "sklearn"
    
    return {
        "status": "healthy",
        "sklearn_available": sklearn_model is not None,
        "gemma_available": gemma_model is not None,
        "default_model": default_model,
        "models_loaded": models_loaded,
        "feature_names": feature_names,
        "classes": label_encoder.classes_.tolist() if label_encoder else []
    }

@app.get("/models", response_model=ModelInfo)
async def get_model_info():
    """Get information about available models."""
    models_loaded = []
    if sklearn_model is not None:
        models_loaded.append("sklearn")
    if gemma_model is not None:
        models_loaded.append("gemma")
    
    default_model = "gemma" if USE_LLM_MODEL and gemma_model else "sklearn"
    
    return ModelInfo(
        sklearn_available=sklearn_model is not None,
        gemma_available=gemma_model is not None,
        default_model=default_model,
        models_loaded=models_loaded
    )

async def predict_with_sklearn(features: IrisFeatures) -> PredictionResponse:
    """Make prediction using sklearn model."""
    if sklearn_model is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Sklearn model not loaded")
    
    # Prepare features
    feature_data = pd.DataFrame({
        'sepal_length': [features.sepal_length],
        'sepal_width': [features.sepal_width],
        'petal_length': [features.petal_length],
        'petal_width': [features.petal_width]
    })
    
    # Make prediction
    prediction = sklearn_model.predict(feature_data)[0]
    probabilities = sklearn_model.predict_proba(feature_data)[0]
    
    # Decode prediction
    species = label_encoder.inverse_transform([prediction])[0]
    confidence = float(max(probabilities))
    
    return PredictionResponse(species=species, confidence=confidence, model_used="sklearn")

async def predict_with_gemma(features: IrisFeatures) -> PredictionResponse:
    """Make prediction using Gemma model."""
    if gemma_model is None:
        raise HTTPException(status_code=503, detail="Gemma model not loaded")
    
    # Make prediction
    species, confidence = gemma_model.predict(
        features.sepal_length, features.sepal_width,
        features.petal_length, features.petal_width
    )
    
    return PredictionResponse(species=species, confidence=confidence, model_used="gemma")

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures, request: Request, model: Optional[str] = None):
    """Predict iris species based on features."""
    
    # Determine which model to use
    if model is None:
        use_gemma = USE_LLM_MODEL and gemma_model is not None
    else:
        if model == "gemma":
            if gemma_model is None:
                raise HTTPException(status_code=400, detail="Gemma model not available")
            use_gemma = True
        elif model == "sklearn":
            if sklearn_model is None:
                raise HTTPException(status_code=400, detail="Sklearn model not available")
            use_gemma = False
        else:
            raise HTTPException(status_code=400, detail="Invalid model type. Use 'sklearn' or 'gemma'")
    
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")
        
        try:
            # Make prediction with chosen model
            if use_gemma:
                result = await predict_with_gemma(features)
            else:
                result = await predict_with_sklearn(features)
            
            latency = round((time.time() - start_time) * 1000, 2)
            
            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "timestamp": time.time(),
                "input": features.model_dump(),
                "result": result.model_dump(),
                "latency_ms": latency,
                "status": "success"
            }))
            
            return result
        
        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "timestamp": time.time(),
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(features_list: List[IrisFeatures], model: Optional[str] = None):
    """Predict iris species for multiple samples."""
    if not features_list:
        return []
    
    try:
        # Make predictions for all samples
        results = []
        for features in features_list:
            result = await predict(features, None, model)  # Reuse single prediction logic
            results.append(result)
        
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

@app.post("/compare_models")
async def compare_models(features: IrisFeatures):
    """Compare predictions from both models on the same input."""
    results = {}
    
    try:
        # Get sklearn prediction
        if sklearn_model is not None:
            sklearn_result = await predict_with_sklearn(features)
            results["sklearn"] = sklearn_result.model_dump()
    except Exception as e:
        results["sklearn"] = {"error": str(e)}
    
    try:
        # Get Gemma prediction
        if gemma_model is not None:
            gemma_result = await predict_with_gemma(features)
            results["gemma"] = gemma_result.model_dump()
    except Exception as e:
        results["gemma"] = {"error": str(e)}
    
    if not results:
        raise HTTPException(status_code=503, detail="No models available for comparison")
    
    return {
        "input": features.model_dump(),
        "predictions": results,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )