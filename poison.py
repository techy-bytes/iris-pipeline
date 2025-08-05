import pandas as pd
import numpy as np
import argparse
import os
from typing import Optional

def poison_labels(input_file: str, output_file: str, poison_rate: float = 0.1, random_seed: int = 42):
    """
    Poison the labels in the iris dataset by randomly changing a percentage of labels.
    
    Args:
        input_file: Path to the original CSV file
        output_file: Path to save the poisoned CSV file
        poison_rate: Percentage of labels to poison (0.0 to 1.0)
        random_seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Read the original dataset
    df = pd.read_csv(input_file)
    
    # Get unique species/labels
    unique_species = df['species'].unique()
    
    # Calculate number of samples to poison
    n_samples = len(df)
    n_poison = int(n_samples * poison_rate)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Unique species: {unique_species}")
    print(f"Poisoning {n_poison} out of {n_samples} samples ({poison_rate*100:.1f}%)")
    
    # Create a copy for poisoning
    poisoned_df = df.copy()
    
    # Randomly select indices to poison
    poison_indices = np.random.choice(n_samples, size=n_poison, replace=False)
    
    # For each selected index, change the label to a different random label
    for idx in poison_indices:
        current_label = poisoned_df.loc[idx, 'species']
        # Choose a different label randomly
        other_labels = [label for label in unique_species if label != current_label]
        new_label = np.random.choice(other_labels)
        poisoned_df.loc[idx, 'species'] = new_label
    
    # Save the poisoned dataset
    poisoned_df.to_csv(output_file, index=False)
    
    print(f"Poisoned dataset saved to: {output_file}")
    print("Original label distribution:")
    print(df['species'].value_counts())
    print("Poisoned label distribution:")
    print(poisoned_df['species'].value_counts())
    
    return output_file

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Poison labels in the iris dataset for adversarial testing',
        epilog='''
Example usage:
  poison.py --rate 0.05                    # Low poisoning (5%%)
  poison.py --rate 0.2 -o corrupted.csv    # Medium poisoning (20%%)
  poison.py --rate 0.4 --seed 123          # High poisoning (40%%) with custom seed
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', default='data/iris.csv', 
                       help='Input CSV file path (default: data/iris.csv)')
    parser.add_argument('--output', '-o', default='data/iris_poisoned.csv',
                       help='Output CSV file path (default: data/iris_poisoned.csv)')
    parser.add_argument('--rate', '-r', type=float, default=0.1,
                       help='Poison rate (0.0 to 1.0, default: 0.1). Recommended: 0.05 (low), 0.1-0.2 (medium), 0.3+ (high)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate poison rate
    if not 0.0 <= args.rate <= 1.0:
        raise ValueError("Poison rate must be between 0.0 and 1.0")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Poison the labels
    poison_labels(args.input, args.output, args.rate, args.seed)

if __name__ == "__main__":
    main()