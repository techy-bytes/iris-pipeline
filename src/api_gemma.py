"""
Gemma LLM API Module for IRIS Classification
This demonstrates the API structure for LLM-based IRIS classification.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
from typing import Optional

app = FastAPI(title="IRIS Gemma LLM API", version="2.0.0")

class IrisLLMRequest(BaseModel):
    sepal_length: float
    sepal_width: float  
    petal_length: float
    petal_width: float

class IrisLLMResponse(BaseModel):
    species: str
    confidence: float
    reasoning: str
    model_type: str

class GemmaLLMPredictor:
    """
    Gemma LLM predictor for IRIS classification.
    In production, this would load the actual fine-tuned Gemma model.
    """
    
    def __init__(self):
        self.model_type = "Gemma-3-1b-it (simulated)"
        self.system_prompt = "Classify the flower based on its measurements into one of the following species: [Setosa, Versicolor, Virginica]"
        self.is_loaded = False
        
    def load_model(self):
        """Load the Gemma model. In production, this would load from GCS."""
        try:
            # In production, this would:
            # 1. Download model from GCS bucket
            # 2. Load tokenizer and model
            # 3. Set up inference pipeline
            print("🔄 Loading Gemma model (simulated)")
            self.is_loaded = True
            print("✅ Gemma model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load Gemma model: {e}")
            return False
    
    def predict(self, features: IrisLLMRequest) -> IrisLLMResponse:
        """Predict species using Gemma LLM."""
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Format input as conversational prompt
        user_content = (
            f"Sepal Length: {features.sepal_length}, "
            f"Sepal Width: {features.sepal_width}, "
            f"Petal Length: {features.petal_length}, "
            f"Petal Width: {features.petal_width}"
        )
        
        # In production, this would:
        # 1. Apply chat template
        # 2. Tokenize input
        # 3. Generate response using the model
        # 4. Decode and parse response
        
        # For now, simulate intelligent prediction based on known IRIS characteristics
        prediction, confidence, reasoning = self._simulate_prediction(features)
        
        return IrisLLMResponse(
            species=prediction,
            confidence=confidence,
            reasoning=reasoning,
            model_type=self.model_type
        )
    
    def _simulate_prediction(self, features: IrisLLMRequest):
        """Simulate Gemma model prediction using rule-based logic."""
        # Simple rules based on IRIS dataset characteristics
        if features.petal_length <= 2.0:
            return "Setosa", 0.95, "Very small petal length indicates Setosa species"
        elif features.petal_length <= 5.0 and features.petal_width <= 1.7:
            return "Versicolor", 0.88, "Medium petal measurements suggest Versicolor species"
        else:
            return "Virginica", 0.92, "Large petal measurements indicate Virginica species"

# Global model instance
gemma_predictor = GemmaLLMPredictor()

@app.on_event("startup")
async def startup_event():
    """Initialize Gemma model on startup."""
    try:
        gemma_predictor.load_model()
    except Exception as e:
        print(f"Warning: Model initialization failed: {e}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "IRIS Gemma LLM API is running",
        "model_type": gemma_predictor.model_type,
        "model_loaded": gemma_predictor.is_loaded
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy" if gemma_predictor.is_loaded else "degraded",
        "model_type": gemma_predictor.model_type,
        "model_loaded": gemma_predictor.is_loaded,
        "api_version": "2.0.0"
    }

@app.post("/predict_llm", response_model=IrisLLMResponse)
async def predict_with_llm(features: IrisLLMRequest):
    """Predict IRIS species using Gemma LLM with reasoning."""
    try:
        result = gemma_predictor.predict(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict", response_model=IrisLLMResponse)
async def predict_legacy(features: IrisLLMRequest):
    """Legacy prediction endpoint (redirects to LLM)."""
    return await predict_with_llm(features)

@app.get("/model_info")
async def model_info():
    """Get information about the current model."""
    return {
        "model_type": gemma_predictor.model_type,
        "model_loaded": gemma_predictor.is_loaded,
        "input_format": "linguistic (conversational)",
        "output_format": "species with reasoning",
        "training_data": "IRIS dataset in conversational format",
        "capabilities": [
            "Species classification",
            "Confidence scoring", 
            "Reasoning explanation",
            "Natural language interaction"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)