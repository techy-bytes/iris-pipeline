import os
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import tempfile
import json
from src.data_transformer import get_linguistic_prediction_prompt

# Optional imports for LLM functionality
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from google.cloud import storage, secretmanager
    from huggingface_hub import login
    LLM_DEPENDENCIES_AVAILABLE = True
except ImportError:
    print("LLM dependencies not available. Using mock implementation only.")
    LLM_DEPENDENCIES_AVAILABLE = False

class GemmaIrisClassifier:
    """LLM-based Iris classifier using fine-tuned Gemma model."""
    
    def __init__(self, 
                 project_id: str = "mlops-466312",
                 bucket_name: str = "week10_unique", 
                 gcs_model_path: str = "output/model-output/",
                 base_model_id: str = "google/gemma-3-1b-it",
                 secret_id: str = "hf_token",
                 use_auth: bool = False):
        
        if not LLM_DEPENDENCIES_AVAILABLE:
            raise ImportError("LLM dependencies not available. Install torch, transformers, peft, google-cloud-storage, and huggingface-hub.")
        
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.gcs_model_path = gcs_model_path
        self.base_model_id = base_model_id
        self.secret_id = secret_id
        self.use_auth = use_auth
        
        self.model = None
        self.tokenizer = None
        self.local_adapter_path = None
        
    def _authenticate_hf(self):
        """Authenticate with Hugging Face using GCP Secret Manager."""
        if not self.use_auth:
            print("Skipping HF authentication (use_auth=False)")
            return
            
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.project_id}/secrets/{self.secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            token = response.payload.data.decode("UTF-8")
            login(token=token)
            print("✅ Successfully logged into Hugging Face.")
        except Exception as e:
            print(f"❌ Failed to authenticate with HF: {e}")
            print("Continuing without authentication...")
    
    def _download_model_from_gcs(self) -> str:
        """Download the fine-tuned model adapter from GCS."""
        try:
            # Create temporary directory for model files
            temp_dir = tempfile.mkdtemp(prefix="gemma_model_")
            
            # Download from GCS
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(self.bucket_name)
            blobs = bucket.list_blobs(prefix=self.gcs_model_path)
            
            downloaded_files = []
            for blob in blobs:
                if not blob.name.endswith('/'):  # Skip directories
                    local_path = os.path.join(temp_dir, 
                                            os.path.relpath(blob.name, self.gcs_model_path))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    blob.download_to_filename(local_path)
                    downloaded_files.append(local_path)
            
            if downloaded_files:
                print(f"✅ Downloaded {len(downloaded_files)} model files to {temp_dir}")
                return temp_dir
            else:
                raise FileNotFoundError(f"No model files found at gs://{self.bucket_name}/{self.gcs_model_path}")
                
        except Exception as e:
            print(f"❌ Failed to download model from GCS: {e}")
            raise
    
    def load_model(self):
        """Load the base model and apply the fine-tuned adapter."""
        try:
            print("Loading Gemma-based Iris classifier...")
            
            # Authenticate with HuggingFace if needed
            self._authenticate_hf()
            
            # Download model from GCS
            self.local_adapter_path = self._download_model_from_gcs()
            
            # Load base model
            print(f"Loading base model: {self.base_model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model with reduced precision for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Apply PEFT adapter
            print("Applying fine-tuned adapter...")
            self.model = PeftModel.from_pretrained(self.model, self.local_adapter_path)
            self.model.eval()
            
            print("✅ Gemma model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Gemma model: {e}")
            return False
    
    def predict(self, sepal_length: float, sepal_width: float, 
                petal_length: float, petal_width: float) -> Tuple[str, float]:
        """Make a prediction using the LLM."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Create linguistic prompt
            messages = get_linguistic_prediction_prompt(
                sepal_length, sepal_width, petal_length, petal_width
            )
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Extract species name and assign confidence
            species = self._extract_species(generated_text)
            confidence = 0.95  # LLM confidence (could be enhanced with actual logits)
            
            return species, confidence
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return "unknown", 0.0
    
    def _extract_species(self, generated_text: str) -> str:
        """Extract species name from generated text."""
        text_lower = generated_text.lower()
        
        if "setosa" in text_lower:
            return "setosa"
        elif "versicolor" in text_lower:
            return "versicolor" 
        elif "virginica" in text_lower:
            return "virginica"
        else:
            # Fallback: return the first word if no exact match
            words = generated_text.split()
            return words[0].lower() if words else "unknown"
    
    def evaluate_on_test_data(self, test_csv_path: str) -> Dict[str, Any]:
        """Evaluate the model on test data and return metrics."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Load test data
        df = pd.read_csv(test_csv_path)
        
        correct = 0
        total = len(df)
        predictions = []
        
        print(f"Evaluating on {total} samples...")
        
        for idx, row in df.iterrows():
            predicted_species, confidence = self.predict(
                row['sepal_length'], row['sepal_width'],
                row['petal_length'], row['petal_width']
            )
            
            actual_species = row['species']
            is_correct = predicted_species.lower() == actual_species.lower()
            
            if is_correct:
                correct += 1
                
            predictions.append({
                'actual': actual_species,
                'predicted': predicted_species,
                'confidence': confidence,
                'correct': is_correct
            })
            
            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{total} samples...")
        
        accuracy = correct / total
        
        metrics = {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'model_type': 'gemma_llm',
            'predictions': predictions
        }
        
        print(f"✅ Evaluation complete: {correct}/{total} correct (accuracy: {accuracy:.4f})")
        
        return metrics
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.local_adapter_path and os.path.exists(self.local_adapter_path):
            import shutil
            shutil.rmtree(self.local_adapter_path)
            print("✅ Cleaned up temporary model files")


class LocalGemmaIrisClassifier:
    """Local version of GemmaIrisClassifier that works with locally trained models."""
    
    def __init__(self, 
                 local_model_path: str = "models/iris-gemma-local",
                 base_model_id: str = "google/gemma-3-1b-it",
                 hf_token: Optional[str] = None):
        
        if not LLM_DEPENDENCIES_AVAILABLE:
            raise ImportError("LLM dependencies not available. Install torch, transformers, peft, and huggingface-hub.")
        
        self.local_model_path = local_model_path
        self.base_model_id = base_model_id
        self.hf_token = hf_token
        
        self.model = None
        self.tokenizer = None
        
    def _authenticate_hf(self):
        """Authenticate with Hugging Face using provided token."""
        if self.hf_token:
            try:
                login(token=self.hf_token)
                print("✅ Successfully logged into Hugging Face.")
            except Exception as e:
                print(f"❌ Failed to authenticate with HF: {e}")
                print("Continuing without authentication...")
        else:
            print("No HF token provided, skipping authentication...")
    
    def load_model(self):
        """Load the locally trained model."""
        try:
            print(f"Loading locally trained Gemma-based Iris classifier from {self.local_model_path}...")
            
            # Check if local model exists
            if not os.path.exists(self.local_model_path):
                raise FileNotFoundError(f"Local model not found at {self.local_model_path}")
            
            # Authenticate with HuggingFace if token provided
            self._authenticate_hf()
            
            # Load base model
            print(f"Loading base model: {self.base_model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model with reduced precision for efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Apply PEFT adapter from local path
            print(f"Applying fine-tuned adapter from {self.local_model_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.local_model_path)
            self.model.eval()
            
            print("✅ Local Gemma model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load local Gemma model: {e}")
            return False
    
    def predict(self, sepal_length: float, sepal_width: float, 
                petal_length: float, petal_width: float) -> Tuple[str, float]:
        """Make a prediction using the local LLM."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Create linguistic prompt
            messages = get_linguistic_prediction_prompt(
                sepal_length, sepal_width, petal_length, petal_width
            )
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Extract species name and assign confidence
            species = self._extract_species(generated_text)
            confidence = 0.95  # LLM confidence (could be enhanced with actual logits)
            
            return species, confidence
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return "unknown", 0.0
    
    def _extract_species(self, generated_text: str) -> str:
        """Extract species name from generated text."""
        text_lower = generated_text.lower()
        
        if "setosa" in text_lower:
            return "setosa"
        elif "versicolor" in text_lower:
            return "versicolor" 
        elif "virginica" in text_lower:
            return "virginica"
        else:
            # Fallback: return the first word if no exact match
            words = generated_text.split()
            return words[0].lower() if words else "unknown"
    
    def evaluate_on_test_data(self, test_csv_path: str) -> Dict[str, Any]:
        """Evaluate the model on test data and return metrics."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Load test data
        df = pd.read_csv(test_csv_path)
        
        correct = 0
        total = len(df)
        predictions = []
        
        print(f"Evaluating on {total} samples...")
        
        for idx, row in df.iterrows():
            predicted_species, confidence = self.predict(
                row['sepal_length'], row['sepal_width'],
                row['petal_length'], row['petal_width']
            )
            
            actual_species = row['species']
            is_correct = predicted_species.lower() == actual_species.lower()
            
            if is_correct:
                correct += 1
                
            predictions.append({
                'actual': actual_species,
                'predicted': predicted_species,
                'confidence': confidence,
                'correct': is_correct
            })
            
            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{total} samples...")
        
        accuracy = correct / total
        
        metrics = {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'model_type': 'local_gemma_llm',
            'predictions': predictions
        }
        
        print(f"✅ Evaluation complete: {correct}/{total} correct (accuracy: {accuracy:.4f})")
        
        return metrics
    
    def cleanup(self):
        """Clean up resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Cleaned up local model resources")


# Mock version for testing without GCS/authentication
class MockGemmaIrisClassifier:
    """Mock version of GemmaIrisClassifier for testing without GCS dependencies."""
    
    def __init__(self, **kwargs):
        self.model_loaded = False
        # Simple rule-based predictions for testing
        self.rules = {
            'setosa': lambda sl, sw, pl, pw: pl < 2.0 and pw < 0.6,
            'versicolor': lambda sl, sw, pl, pw: 2.0 <= pl < 5.0 and 0.6 <= pw < 1.8,
            'virginica': lambda sl, sw, pl, pw: pl >= 5.0 or pw >= 1.8
        }
    
    def load_model(self):
        """Mock model loading."""
        print("Loading mock Gemma classifier...")
        self.model_loaded = True
        return True
    
    def predict(self, sepal_length: float, sepal_width: float, 
                petal_length: float, petal_width: float) -> Tuple[str, float]:
        """Make a mock prediction using simple rules."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Apply rules in order
        for species, rule in self.rules.items():
            if rule(sepal_length, sepal_width, petal_length, petal_width):
                return species, 0.85
        
        return "versicolor", 0.85  # default
    
    def evaluate_on_test_data(self, test_csv_path: str) -> Dict[str, Any]:
        """Mock evaluation."""
        df = pd.read_csv(test_csv_path)
        correct = 0
        total = len(df)
        
        for _, row in df.iterrows():
            predicted_species, _ = self.predict(
                row['sepal_length'], row['sepal_width'],
                row['petal_length'], row['petal_width']
            )
            if predicted_species.lower() == row['species'].lower():
                correct += 1
        
        accuracy = correct / total
        return {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'model_type': 'mock_gemma_llm'
        }
    
    def cleanup(self):
        """Mock cleanup."""
        pass