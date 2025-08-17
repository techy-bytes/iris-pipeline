#!/usr/bin/env python3
"""
Demo script showcasing the enhanced Iris pipeline with Gemma LLM integration.
Run this to see model comparison in action.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train_enhanced import compare_models
from src.data_transformer import convert_iris_to_linguistic_format
import pandas as pd

def main():
    print("🌸 IRIS CLASSIFICATION - GEMMA LLM INTEGRATION DEMO")
    print("=" * 60)
    
    # Step 1: Show original data
    print("\n1. ORIGINAL IRIS DATASET:")
    df = pd.read_csv('data/iris.csv')
    print(f"   Shape: {df.shape}")
    print(f"   Sample data:")
    print(df.head(3).to_string(index=False))
    
    # Step 2: Convert to linguistic format
    print("\n2. CONVERTING TO LINGUISTIC FORMAT:")
    train_count, eval_count = convert_iris_to_linguistic_format(
        'data/iris.csv', 'data/train.jsonl', 'data/eval.jsonl'
    )
    
    # Show sample linguistic data
    import json
    with open('data/train.jsonl', 'r') as f:
        sample = json.loads(f.readline())
    
    print("\n   Sample linguistic format:")
    print(f"   System: {sample['messages'][0]['content']}")
    print(f"   User:   {sample['messages'][1]['content']}")
    print(f"   Assistant: {sample['messages'][2]['content']}")
    
    # Step 3: Compare models
    print("\n3. MODEL COMPARISON:")
    results = compare_models()
    
    # Step 4: Show metrics
    print("\n4. FINAL METRICS:")
    if os.path.exists('metrics.csv'):
        metrics_df = pd.read_csv('metrics.csv')
        print(metrics_df.to_string(index=False))
    else:
        print("   Metrics file not found - using comparison results")
    
    # Step 5: Recommendation
    print("\n5. RECOMMENDATION:")
    winner = results['winner']
    improvement = results['accuracy_difference']
    
    if winner == 'gemma':
        print(f"   ✅ Use Gemma LLM model (improvement: +{improvement:.1%})")
        print("   📋 Next steps:")
        print("   - Train real Gemma model using mlops-w10.ipynb on Kaggle")
        print("   - Upload to gs://week10_unique/output/model-output/")
        print("   - Set USE_LLM_MODEL=true, USE_MOCK_LLM=false in production")
    elif winner == 'sklearn':
        print(f"   ✅ Sklearn model performs better (by {improvement:.1%})")
        print("   📋 Continue using traditional ML approach")
    else:
        print("   🤝 Both models perform equally well")
    
    print("\n6. INTEGRATION COMPLETE! 🎉")
    print("   See GEMMA_INTEGRATION.md for full documentation")

if __name__ == "__main__":
    main()