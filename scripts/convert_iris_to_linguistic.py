#!/usr/bin/env python3
"""
Convert numerical IRIS dataset to linguistic/conversational format for LLM training.
"""

import pandas as pd
import json
import os
from pathlib import Path

def convert_iris_to_linguistic(input_path='data/iris.csv', output_dir='data'):
    """
    Convert IRIS dataset from numerical features to conversational format.
    
    Input format: sepal_length,sepal_width,petal_length,petal_width,species
    Output format: JSON with conversational messages for each sample
    """
    # Load the original IRIS dataset
    df = pd.read_csv(input_path)
    
    # Clean column names (remove BOM if present)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    print(f"Loaded IRIS dataset with {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"Species distribution:\n{df['species'].value_counts()}")
    
    # Prepare conversational data
    train_data = []
    eval_data = []
    
    for idx, row in df.iterrows():
        # Create descriptive text from measurements
        user_content = (
            f"Sepal Length: {row['sepal_length']}, "
            f"Sepal Width: {row['sepal_width']}, "
            f"Petal Length: {row['petal_length']}, "
            f"Petal Width: {row['petal_width']}"
        )
        
        # Capitalize species name for consistency
        species = row['species'].capitalize()
        
        # Create conversation in chat format
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": "Classify the flower based on its measurements into one of the following species: [Setosa, Versicolor, Virginica]"
                },
                {
                    "role": "user", 
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": species
                }
            ]
        }
        
        # Split data: 80% train, 20% eval
        if idx % 5 == 0:  # Every 5th sample goes to eval
            eval_data.append(conversation)
        else:
            train_data.append(conversation)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    train_file = os.path.join(output_dir, 'train.jsonl')
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Save evaluation data
    eval_file = os.path.join(output_dir, 'eval.jsonl')
    with open(eval_file, 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nConversion completed!")
    print(f"Training samples: {len(train_data)} -> {train_file}")
    print(f"Evaluation samples: {len(eval_data)} -> {eval_file}")
    
    # Create a small sample for testing
    sample_file = os.path.join(output_dir, 'iris_sample.json')
    
    # Find samples for each species by checking the assistant content
    setosa_sample = None
    versicolor_sample = None  
    virginica_sample = None
    
    for item in train_data:
        assistant_content = item["messages"][2]["content"]  # assistant response
        if assistant_content == "Setosa" and setosa_sample is None:
            setosa_sample = item
        elif assistant_content == "Versicolor" and versicolor_sample is None:
            versicolor_sample = item
        elif assistant_content == "Virginica" and virginica_sample is None:
            virginica_sample = item
        
        if setosa_sample and versicolor_sample and virginica_sample:
            break
    
    sample_data = {
        "setosa_sample": setosa_sample,
        "versicolor_sample": versicolor_sample,
        "virginica_sample": virginica_sample,
    }
    
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample data: {sample_file}")
    
    return train_file, eval_file, sample_file

if __name__ == "__main__":
    convert_iris_to_linguistic()