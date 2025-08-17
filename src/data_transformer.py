import pandas as pd
import json
import os
from typing import List, Dict, Any


def convert_iris_to_linguistic_format(input_csv_path: str, output_train_path: str, output_eval_path: str, train_split: float = 0.8):
    """Convert numeric Iris data to linguistic format for LLM training."""
    
    # Load the original iris dataset
    df = pd.read_csv(input_csv_path)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/eval
    train_size = int(len(df) * train_split)
    train_df = df[:train_size]
    eval_df = df[train_size:]
    
    def create_linguistic_examples(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame rows to linguistic chat format."""
        examples = []
        
        for _, row in dataframe.iterrows():
            # Create natural language description of the measurements
            user_content = (
                f"Sepal Length: {row['sepal_length']}, Sepal Width: {row['sepal_width']}, "
                f"Petal Length: {row['petal_length']}, Petal Width: {row['petal_width']}"
            )
            
            # Create the chat format expected by the LLM
            messages = [
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
                    "content": row['species'].capitalize()
                }
            ]
            
            examples.append({"messages": messages})
        
        return examples
    
    # Create linguistic examples
    train_examples = create_linguistic_examples(train_df)
    eval_examples = create_linguistic_examples(eval_df)
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_train_path) if os.path.dirname(output_train_path) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_eval_path) if os.path.dirname(output_eval_path) else '.', exist_ok=True)
    
    # Save as JSONL files
    with open(output_train_path, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    with open(output_eval_path, 'w') as f:
        for example in eval_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Created linguistic training data:")
    print(f"  Training examples: {len(train_examples)} -> {output_train_path}")
    print(f"  Evaluation examples: {len(eval_examples)} -> {output_eval_path}")
    
    return len(train_examples), len(eval_examples)


def get_linguistic_prediction_prompt(sepal_length: float, sepal_width: float, 
                                   petal_length: float, petal_width: float) -> str:
    """Create a prediction prompt in the same format as training data."""
    user_content = (
        f"Sepal Length: {sepal_length}, Sepal Width: {sepal_width}, "
        f"Petal Length: {petal_length}, Petal Width: {petal_width}"
    )
    
    messages = [
        {
            "role": "system", 
            "content": "Classify the flower based on its measurements into one of the following species: [Setosa, Versicolor, Virginica]"
        },
        {
            "role": "user", 
            "content": user_content
        }
    ]
    
    return messages


if __name__ == "__main__":
    # Create the linguistic format data files
    input_path = "data/iris.csv"
    train_output = "data/train.jsonl"
    eval_output = "data/eval.jsonl"
    
    convert_iris_to_linguistic_format(input_path, train_output, eval_output)