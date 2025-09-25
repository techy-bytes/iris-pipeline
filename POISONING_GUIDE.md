# Label Poisoning Guide

This guide explains how to use different levels of label poisoning for the iris dataset and their typical use cases.

## ⚠️ Security Alert: Automatic Poisoning Detection

**IMPORTANT**: The pipeline now includes automatic data poisoning detection that will **fail builds and tests** when poisoning is detected. This is a security feature to prevent unintentional deployment of models trained on corrupted data.

- ✅ **Safe**: Original dataset (`data/iris.csv`) - builds pass
- 🚨 **Unsafe**: Poisoned dataset (`data/iris_poisoned.csv`) - builds fail with security alert

## How Label Poisoning Works

The `poison.py` script randomly selects a percentage of samples from the dataset and changes their labels to different, incorrect labels. For example:
- A sample originally labeled as "setosa" might be randomly changed to "versicolor" or "virginica"
- The feature data (sepal length, sepal width, etc.) remains unchanged
- Only the species labels are modified

## Poisoning Levels and Use Cases

### Low Poisoning (1-10%)
- **Rate**: `--rate 0.01` to `--rate 0.1`
- **Use Case**: Simulate real-world labeling errors, test model robustness
- **Impact**: Minimal visible change in label distribution
- **Example**: `python poison.py --rate 0.05` (5% poisoning)

### Medium Poisoning (10-30%)
- **Rate**: `--rate 0.1` to `--rate 0.3`
- **Use Case**: Test model performance under moderate adversarial conditions
- **Impact**: Noticeable imbalance in label distribution
- **Example**: `python poison.py --rate 0.2` (20% poisoning)

### High Poisoning (30-50%)
- **Rate**: `--rate 0.3` to `--rate 0.5`
- **Use Case**: Stress testing, adversarial research, worst-case scenarios
- **Impact**: Significant corruption, major distribution shifts
- **Example**: `python poison.py --rate 0.4` (40% poisoning)

### Extreme Poisoning (50%+)
- **Rate**: `--rate 0.5` to `--rate 1.0`
- **Use Case**: Academic research on label noise tolerance
- **Impact**: Dataset becomes heavily corrupted, may be unusable
- **Example**: `python poison.py --rate 0.7` (70% poisoning)

## How to Change Poisoning Levels

### Command Line Usage
```bash
# Low poisoning (5%)
python poison.py --rate 0.05 --output data/iris_low_poison.csv

# Medium poisoning (20%)
python poison.py --rate 0.2 --output data/iris_medium_poison.csv

# High poisoning (40%)
python poison.py --rate 0.4 --output data/iris_high_poison.csv

# Custom seed for reproducibility
python poison.py --rate 0.15 --seed 123 --output data/iris_custom.csv
```

### CI/CD Pipeline Configuration
To change the poisoning level in the CI/CD pipeline, edit `.github/workflows/ci.yml`:

```yaml
- name: Generate poisoned dataset
  run: |
    echo -e "\n## Data Poisoning Report" >> report.md
    python poison.py --rate 0.15 --output data/iris_poisoned.csv >> report.md 2>&1
```

## Testing Different Levels

### Iterative Testing Approach
1. **Start Low**: Begin with 5-10% poisoning to establish baseline impact
2. **Increment Gradually**: Test 10%, 20%, 30% to see degradation patterns
3. **Document Results**: Track model accuracy vs poisoning rate
4. **Find Threshold**: Identify the poisoning level where performance becomes unacceptable

### Batch Testing Script Example
```bash
#!/bin/bash
# Test multiple poisoning levels
for rate in 0.05 0.1 0.15 0.2 0.25 0.3; do
    echo "Testing poisoning rate: $rate"
    python poison.py --rate $rate --output "data/iris_poison_${rate}.csv"
    # Add your model training/testing commands here
done
```

## Recommended Testing Strategy

1. **Development**: Use 10% poisoning (default) for regular testing
2. **Model Validation**: Test at 5%, 15%, 25% to create robustness curves
3. **Production Testing**: Use 5% to simulate realistic label errors
4. **Research**: Explore 30%+ for adversarial machine learning studies

## Output Analysis

Each run provides detailed statistics:
- Original vs poisoned label distributions
- Number of samples modified
- Exact poisoning percentage achieved

Monitor these metrics to understand how poisoning affects your dataset balance and downstream model performance.

## Security Features

### Automatic Poisoning Detection
The pipeline includes built-in security measures that automatically detect data poisoning:

```python
# Security test that fails when poisoning is detected
def test_no_data_poisoning_detected():
    """Critical security test: Fail build if data poisoning is detected."""
    if os.path.exists('data/iris_poisoned.csv'):
        pytest.fail("🚨 CRITICAL SECURITY ALERT: DATA POISONING DETECTED 🚨")
```

### Security Alerts
When poisoning is detected, you'll see detailed failure messages:
```
🚨 CRITICAL SECURITY ALERT: DATA POISONING DETECTED 🚨
Poisoned dataset found at: data/iris_poisoned.csv
Poisoning rate: 10.0% (15/150 samples)
Original distribution: {'setosa': 50, 'versicolor': 50, 'virginica': 50}
Poisoned distribution: {'virginica': 52, 'setosa': 51, 'versicolor': 47}
BUILD FAILED for security reasons. Remove poisoned dataset to proceed.
```

### Working with Poisoning for Research
If you need to work with poisoned data for research purposes:

1. **Use different output filenames**: Avoid `data/iris_poisoned.csv`
2. **Temporary testing**: Use `data/temp_poisoned.csv` or similar
3. **Clean up**: Always remove poisoned datasets before committing
4. **CI/CD**: The pipeline will block deployment if poisoning is detected

Example for research testing:
```bash
# Safe - won't trigger security alerts
python poison.py --rate 0.1 --output data/research_poisoned.csv
python src/train.py  # Will still use original data/iris.csv

# Unsafe - will fail builds
python poison.py --rate 0.1 --output data/iris_poisoned.csv
python -m pytest  # FAILS with security alert
```