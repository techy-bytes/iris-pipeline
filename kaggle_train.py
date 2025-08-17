#!/usr/bin/env python3
"""
Kaggle Local Training Script for Iris Classification with Gemma LLM

This script trains a Gemma-3-1b-it model locally using Kaggle's P100 GPU
for Iris flower classification. It converts the numeric Iris data to 
linguistic format and fine-tunes the model using PEFT/LoRA adapters.

Usage:
    # Set up Kaggle environment
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
    export HF_TOKEN=your_huggingface_token
    
    # Run training
    python kaggle_train.py

Requirements:
    - Kaggle account with P100 GPU access
    - HuggingFace account and token
    - Local installation of this repository
"""

import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import json
from typing import Dict, List
from src.data_transformer import DataTransformer
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleIrisTrainer:
    """Trainer for Gemma model on Iris classification using Kaggle's infrastructure."""
    
    def __init__(self, 
                 base_model_id: str = "google/gemma-3-1b-it",
                 output_dir: str = "models/iris-gemma-local",
                 hf_token: str = None):
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        self.hf_token = hf_token
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        
    def setup_environment(self):
        """Setup the training environment."""
        logger.info("Setting up training environment...")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Training will be slow on CPU.")
        else:
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
            
        # Login to HuggingFace
        if self.hf_token:
            from huggingface_hub import login
            login(token=self.hf_token)
            logger.info("✅ Logged into HuggingFace")
        else:
            logger.warning("No HF token provided. Some models may not be accessible.")
    
    def prepare_data(self):
        """Prepare the linguistic training data."""
        logger.info("Preparing linguistic training data...")
        
        # Load and transform data
        transformer = DataTransformer()
        
        # Load original iris data
        iris_df = pd.read_csv('data/iris.csv')
        logger.info(f"Loaded {len(iris_df)} samples from iris.csv")
        
        # Convert to linguistic format
        transformer.create_linguistic_dataset(iris_df, 'data/train_local.jsonl')
        logger.info("Created linguistic training data")
        
        # Load the linguistic data
        with open('data/train_local.jsonl', 'r') as f:
            data = [json.loads(line) for line in f]
        
        # Format for training (extract text from messages)
        formatted_data = []
        for item in data:
            messages = item['messages']
            # Convert messages to a single text string
            text = self._format_messages_for_training(messages)
            formatted_data.append({"text": text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        return dataset
    
    def _format_messages_for_training(self, messages: List[Dict]) -> str:
        """Format messages for training as a single text string."""
        formatted_parts = []
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                formatted_parts.append(f"<system>{content}</system>")
            elif role == 'user':
                formatted_parts.append(f"<user>{content}</user>")
            elif role == 'assistant':
                formatted_parts.append(f"<assistant>{content}</assistant>")
        
        return "\n".join(formatted_parts)
    
    def load_model(self):
        """Load and prepare the base model for training."""
        logger.info(f"Loading base model: {self.base_model_id}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("✅ Model loaded and prepared for training")
    
    def setup_lora_config(self):
        """Setup LoRA configuration for efficient fine-tuning."""
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Alpha parameter
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("✅ LoRA configuration applied")
    
    def train(self, dataset, epochs: int = 3, batch_size: int = 4):
        """Train the model using SFT (Supervised Fine-Tuning)."""
        logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}")
        
        # Training arguments optimized for P100 GPU
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="no",
            save_strategy="epoch",
            learning_rate=2e-4,
            fp16=True,  # Use mixed precision for P100
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to="none"  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            args=training_args,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            max_seq_length=512,
            dataset_text_field="text"
        )
        
        # Train the model
        logger.info("🚀 Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"✅ Training complete! Model saved to {self.output_dir}")
    
    def test_model(self):
        """Test the trained model with a sample prediction."""
        logger.info("Testing trained model...")
        
        try:
            from src.gemma_classifier import LocalGemmaIrisClassifier
            
            # Load the trained model
            classifier = LocalGemmaIrisClassifier(
                local_model_path=self.output_dir,
                hf_token=self.hf_token
            )
            
            if classifier.load_model():
                # Test prediction
                species, confidence = classifier.predict(5.1, 3.5, 1.4, 0.2)
                logger.info(f"Test prediction: {species} (confidence: {confidence:.3f})")
                
                # Evaluate on test data if available
                if os.path.exists('data/iris.csv'):
                    metrics = classifier.evaluate_on_test_data('data/iris.csv')
                    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
                
                classifier.cleanup()
                logger.info("✅ Model test successful!")
            else:
                logger.error("❌ Failed to load trained model")
                
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Gemma model locally for Iris classification")
    parser.add_argument("--output_dir", default="models/iris-gemma-local", 
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Training batch size")
    parser.add_argument("--hf_token", default=None, 
                       help="HuggingFace token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Get HF token from args or environment
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    
    if not hf_token:
        logger.warning("No HuggingFace token provided. Some models may not be accessible.")
    
    # Create trainer
    trainer = KaggleIrisTrainer(
        output_dir=args.output_dir,
        hf_token=hf_token
    )
    
    try:
        # Setup environment
        trainer.setup_environment()
        
        # Prepare data
        dataset = trainer.prepare_data()
        
        # Load and setup model
        trainer.load_model()
        trainer.setup_lora_config()
        
        # Train
        trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size)
        
        # Test the trained model
        trainer.test_model()
        
        logger.info("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()