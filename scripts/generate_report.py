#!/usr/bin/env python3
"""
Generate markdown report from Gemma evaluation results
"""
import json
import sys

def generate_detailed_results():
    try:
        with open('gemma_evaluation_results.json', 'r') as f:
            results = json.load(f)
        
        print('### Detailed Evaluation Results')
        print()
        print('#### Base Gemma 3 Model Performance:')
        base_metrics = results['base_model_metrics']
        print('- **Accuracy**: {:.4f}'.format(base_metrics['accuracy']))
        print('- **Precision**: {:.4f}'.format(base_metrics['precision']))
        print('- **Recall**: {:.4f}'.format(base_metrics['recall']))
        print('- **F1-Score**: {:.4f}'.format(base_metrics['f1_score']))
        print('- **Average Confidence**: {:.4f}'.format(base_metrics['average_confidence']))
        print()
        print('#### Fine-tuned Gemma 3 Model Performance:')
        ft_metrics = results['finetuned_model_metrics']
        print('- **Accuracy**: {:.4f}'.format(ft_metrics['accuracy']))
        print('- **Precision**: {:.4f}'.format(ft_metrics['precision']))
        print('- **Recall**: {:.4f}'.format(ft_metrics['recall']))
        print('- **F1-Score**: {:.4f}'.format(ft_metrics['f1_score']))
        print('- **Average Confidence**: {:.4f}'.format(ft_metrics['average_confidence']))
        print()
        print('#### Performance Improvements:')
        improvements = results['improvements']
        print('- **Accuracy Improvement**: {:+.2f}%'.format(improvements['accuracy_improvement_percent']))
        print('- **Precision Improvement**: {:+.2f}%'.format(improvements['precision_improvement_percent']))
        print('- **Recall Improvement**: {:+.2f}%'.format(improvements['recall_improvement_percent']))
        print('- **F1-Score Improvement**: {:+.2f}%'.format(improvements['f1_improvement_percent']))
        print()
        status = 'PASSED' if results.get('validation_passed', False) else 'FAILED'
        print('#### Validation Status: {}'.format(status))
        print('- **Training Time**: {:.2f} seconds'.format(results.get('training_time_seconds', 0)))
        print('- **Test Samples**: {}'.format(results['dataset_info']['test_samples']))
        print('- **Training Samples**: {}'.format(results['dataset_info']['training_samples']))
        print('- **Total Dataset Size**: {}'.format(results['dataset_info']['total_samples']))
        
    except FileNotFoundError:
        print('⚠️ Gemma evaluation results not found')
    except Exception as e:
        print('❌ Error generating detailed results: {}'.format(str(e)))

def generate_summary():
    try:
        with open('gemma_evaluation_results.json', 'r') as f:
            results = json.load(f)
        
        status = 'PASSED' if results.get('validation_passed', False) else 'FAILED'
        print('Gemma Model Evaluation: {}'.format(status))
        print('Accuracy Improvement: {:+.2f}%'.format(results['improvements']['accuracy_improvement_percent']))
        print('F1-Score Improvement: {:+.2f}%'.format(results['improvements']['f1_improvement_percent']))
        
    except FileNotFoundError:
        print('⚠️ Gemma evaluation results not found')
    except Exception as e:
        print('❌ Error generating summary: {}'.format(str(e)))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        generate_summary()
    else:
        generate_detailed_results()
