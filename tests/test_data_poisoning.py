import pandas as pd
import pytest
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from poison import poison_labels


@pytest.fixture(scope="module")
def original_data():
    """Fixture to load the original iris dataset."""
    data_path = 'data/iris.csv'
    if not os.path.exists(data_path):
        pytest.fail(f"Original data file not found at {data_path}")
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def poisoned_data():
    """Fixture to load the poisoned iris dataset."""
    data_path = 'data/iris_poisoned.csv'
    if not os.path.exists(data_path):
        pytest.fail(f"Poisoned data file not found at {data_path}")
    return pd.read_csv(data_path)


class TestWorkflowPoisoningIntegration:
    """Test suite for validating poisoning detection in the ML workflow."""
    
    def test_workflow_can_detect_poisoned_data(self, original_data, poisoned_data):
        """Test that the workflow can detect when poisoned data is being used."""
        # This test simulates what a production system might do to detect poisoning
        
        # Calculate metrics that indicate poisoning
        different_labels = (original_data['species'] != poisoned_data['species']).sum()
        total_samples = len(original_data)
        detected_poison_rate = different_labels / total_samples
        
        # Check if we can reliably detect poisoning occurred
        assert detected_poison_rate > 0, "Workflow failed to detect any data poisoning"
        
        # Validate the detection is within expected bounds
        assert 0.01 <= detected_poison_rate <= 0.5, (
            f"Detected poisoning rate ({detected_poison_rate:.3f}) is outside expected workflow bounds"
        )
        
        print(f"✓ Workflow successfully detected {detected_poison_rate:.1%} data poisoning")
    
    def test_workflow_poisoning_impact_monitoring(self, original_data, poisoned_data):
        """Test that the workflow can monitor the impact of poisoning on performance."""
        # This test validates that we can measure performance degradation due to poisoning
        
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        # Train on original data
        X_orig = original_data[feature_cols]
        y_orig = original_data['species']
        le = LabelEncoder()
        y_orig_encoded = le.fit_transform(y_orig)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_orig, y_orig_encoded, test_size=0.3, random_state=42, stratify=y_orig_encoded
        )
        
        model_orig = RandomForestClassifier(random_state=42, n_estimators=100)
        model_orig.fit(X_train, y_train)
        baseline_accuracy = accuracy_score(y_test, model_orig.predict(X_test))
        
        # Train on poisoned data
        X_poison = poisoned_data[feature_cols]
        y_poison = poisoned_data['species']
        y_poison_encoded = le.fit_transform(y_poison)
        
        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X_poison, y_poison_encoded, test_size=0.3, random_state=42, stratify=y_poison_encoded
        )
        
        model_poison = RandomForestClassifier(random_state=42, n_estimators=100)
        model_poison.fit(X_train_p, y_train_p)
        poisoned_accuracy = accuracy_score(y_test_p, model_poison.predict(X_test_p))
        
        # Calculate impact metrics that a monitoring system would track
        accuracy_drop = baseline_accuracy - poisoned_accuracy
        relative_drop = accuracy_drop / baseline_accuracy
        
        # Validate we can measure meaningful impact
        assert accuracy_drop > 0, "Workflow failed to detect accuracy degradation from poisoning"
        assert relative_drop < 0.3, f"Accuracy drop ({relative_drop:.1%}) is too severe - check poisoning level"
        
        print(f"✓ Workflow detected {accuracy_drop:.3f} accuracy drop ({relative_drop:.1%}) due to poisoning")


class TestDataPoisoningDetection:
    """Test suite for detecting data poisoning in datasets."""
    
    def test_poisoning_exists(self, original_data, poisoned_data):
        """Verify that poisoning has been applied to the dataset."""
        # Compare labels between original and poisoned datasets
        different_labels = (original_data['species'] != poisoned_data['species']).sum()
        assert different_labels > 0, "No poisoning detected - datasets have identical labels"
    
    def test_poisoning_rate_reasonable(self, original_data, poisoned_data):
        """Verify that the poisoning rate is within reasonable bounds."""
        different_labels = (original_data['species'] != poisoned_data['species']).sum()
        total_samples = len(original_data)
        poison_rate = different_labels / total_samples
        
        # Expect poisoning rate between 1% and 50%
        assert 0.01 <= poison_rate <= 0.5, f"Poisoning rate ({poison_rate:.3f}) is outside reasonable bounds (0.01-0.5)"
    
    def test_features_unchanged(self, original_data, poisoned_data):
        """Verify that only labels were changed, not feature data."""
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        for col in feature_cols:
            # Features should be identical between original and poisoned datasets
            assert original_data[col].equals(poisoned_data[col]), f"Feature column '{col}' was modified during poisoning"
    
    def test_dataset_size_preserved(self, original_data, poisoned_data):
        """Verify that dataset size remains the same after poisoning."""
        assert len(original_data) == len(poisoned_data), "Dataset size changed during poisoning"
        assert original_data.shape[1] == poisoned_data.shape[1], "Number of columns changed during poisoning"


class TestPoisoningScriptValidation:
    """Test suite for validating the poison.py script functionality."""
    
    def test_poison_script_basic_functionality(self, tmp_path):
        """Test that the poison script works with basic parameters."""
        # Create a temporary test dataset with multiple species
        test_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 4.6],
            'sepal_width': [3.5, 3.0, 3.2, 3.1],
            'petal_length': [1.4, 1.4, 1.3, 1.5],
            'petal_width': [0.2, 0.2, 0.2, 0.2],
            'species': ['setosa', 'versicolor', 'setosa', 'virginica']
        })
        
        input_file = tmp_path / "test_input.csv"
        output_file = tmp_path / "test_output.csv"
        test_data.to_csv(input_file, index=False)
        
        # Apply poisoning
        poison_labels(str(input_file), str(output_file), poison_rate=0.5, random_seed=42)
        
        # Verify output file was created
        assert output_file.exists(), "Poison script did not create output file"
        
        # Load and verify poisoned data
        poisoned = pd.read_csv(output_file)
        assert len(poisoned) == len(test_data), "Poisoned dataset has different size"
        
        # With 50% poisoning on 4 samples, expect 2 samples to be poisoned
        different_labels = (test_data['species'] != poisoned['species']).sum()
        assert different_labels == 2, f"Expected 2 poisoned samples, got {different_labels}"
    
    def test_poison_script_zero_rate(self, tmp_path):
        """Test that poison script with 0% rate doesn't change anything."""
        test_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9],
            'sepal_width': [3.5, 3.0],
            'petal_length': [1.4, 1.4],
            'petal_width': [0.2, 0.2],
            'species': ['setosa', 'versicolor']
        })
        
        input_file = tmp_path / "test_input.csv"
        output_file = tmp_path / "test_output.csv"
        test_data.to_csv(input_file, index=False)
        
        # Apply 0% poisoning
        poison_labels(str(input_file), str(output_file), poison_rate=0.0, random_seed=42)
        
        poisoned = pd.read_csv(output_file)
        assert test_data.equals(poisoned), "0% poisoning should not change the dataset"
    
    def test_poison_script_full_rate(self, tmp_path):
        """Test that poison script with 100% rate changes all possible labels."""
        test_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7],
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'versicolor', 'virginica']
        })
        
        input_file = tmp_path / "test_input.csv"
        output_file = tmp_path / "test_output.csv"
        test_data.to_csv(input_file, index=False)
        
        # Apply 100% poisoning
        poison_labels(str(input_file), str(output_file), poison_rate=1.0, random_seed=42)
        
        poisoned = pd.read_csv(output_file)
        different_labels = (test_data['species'] != poisoned['species']).sum()
        assert different_labels == 3, f"Expected all 3 samples to be poisoned, got {different_labels}"


class TestPoisoningImpactMeasurement:
    """Test suite for measuring the impact of poisoning on model performance."""
    
    def test_poisoning_reduces_accuracy(self, original_data, poisoned_data):
        """Verify that poisoning reduces model accuracy."""
        # Train model on original data
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        # Original model
        X_orig = original_data[feature_cols]
        y_orig = original_data['species']
        le = LabelEncoder()
        y_orig_encoded = le.fit_transform(y_orig)
        
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X_orig, y_orig_encoded, test_size=0.3, random_state=42, stratify=y_orig_encoded
        )
        
        model_orig = RandomForestClassifier(random_state=42, n_estimators=50)
        model_orig.fit(X_train_orig, y_train_orig)
        y_pred_orig = model_orig.predict(X_test_orig)
        accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)
        
        # Poisoned model
        X_poison = poisoned_data[feature_cols]
        y_poison = poisoned_data['species']
        y_poison_encoded = le.fit_transform(y_poison)
        
        X_train_poison, X_test_poison, y_train_poison, y_test_poison = train_test_split(
            X_poison, y_poison_encoded, test_size=0.3, random_state=42, stratify=y_poison_encoded
        )
        
        model_poison = RandomForestClassifier(random_state=42, n_estimators=50)
        model_poison.fit(X_train_poison, y_train_poison)
        y_pred_poison = model_poison.predict(X_test_poison)
        accuracy_poison = accuracy_score(y_test_poison, y_pred_poison)
        
        # Accuracy should be lower with poisoned data
        assert accuracy_poison < accuracy_orig, (
            f"Poisoned model accuracy ({accuracy_poison:.4f}) should be lower than "
            f"original model accuracy ({accuracy_orig:.4f})"
        )
        
        # Accuracy drop should be reasonable (not catastrophic unless poisoning rate is very high)
        accuracy_drop = accuracy_orig - accuracy_poison
        assert accuracy_drop < 0.5, f"Accuracy drop ({accuracy_drop:.4f}) is too large - check poisoning level"
    
    def test_accuracy_degradation_threshold(self, original_data, poisoned_data):
        """Test that accuracy degradation is within acceptable bounds for the current poisoning level."""
        # Calculate actual poisoning rate
        different_labels = (original_data['species'] != poisoned_data['species']).sum()
        poison_rate = different_labels / len(original_data)
        
        # For low poisoning rates (< 10%), accuracy should not drop below 80%
        if poison_rate < 0.1:
            feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            X = poisoned_data[feature_cols]
            y = poisoned_data['species']
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
            
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            assert accuracy >= 0.8, (
                f"With low poisoning rate ({poison_rate:.3f}), "
                f"accuracy ({accuracy:.4f}) should be at least 0.8"
            )


class TestPoisoningDistribution:
    """Test suite for validating how poisoning is distributed across classes."""
    
    def test_all_classes_present_after_poisoning(self, poisoned_data):
        """Verify that all original classes are still present after poisoning."""
        expected_classes = {'setosa', 'versicolor', 'virginica'}
        actual_classes = set(poisoned_data['species'].unique())
        
        assert actual_classes == expected_classes, (
            f"Missing classes after poisoning. Expected {expected_classes}, got {actual_classes}"
        )
    
    def test_poisoning_affects_multiple_classes(self, original_data, poisoned_data):
        """Verify that poisoning doesn't unfairly target a single class."""
        # Count changes per original class
        changes_per_class = {}
        for orig_class in original_data['species'].unique():
            mask = original_data['species'] == orig_class
            orig_subset = original_data[mask]['species']
            poison_subset = poisoned_data[mask]['species']
            changes = (orig_subset != poison_subset).sum()
            changes_per_class[orig_class] = changes
        
        total_changes = sum(changes_per_class.values())
        
        # If there are multiple changes, they should be distributed across classes
        if total_changes > 2:
            classes_with_changes = sum(1 for changes in changes_per_class.values() if changes > 0)
            assert classes_with_changes >= 2, (
                f"Poisoning should affect multiple classes, but only {classes_with_changes} "
                f"classes were affected: {changes_per_class}"
            )
    
    def test_class_balance_not_severely_disrupted(self, original_data, poisoned_data):
        """Verify that poisoning doesn't severely disrupt class balance."""
        orig_counts = original_data['species'].value_counts()
        poison_counts = poisoned_data['species'].value_counts()
        
        for class_name in orig_counts.index:
            orig_count = orig_counts[class_name]
            poison_count = poison_counts[class_name]
            
            # No class should lose more than 20% of its samples or gain more than 20%
            change_ratio = abs(poison_count - orig_count) / orig_count
            assert change_ratio <= 0.2, (
                f"Class '{class_name}' balance severely disrupted: "
                f"original {orig_count}, poisoned {poison_count} (change: {change_ratio:.2%})"
            )


class TestPoisoningFileIntegrity:
    """Test suite for validating file integrity and format consistency."""
    
    def test_poisoned_file_format_matches_original(self, original_data, poisoned_data):
        """Verify that poisoned file maintains the same format as original."""
        # Same columns
        assert list(original_data.columns) == list(poisoned_data.columns), (
            "Column names differ between original and poisoned datasets"
        )
        
        # Same data types for feature columns
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in feature_cols:
            assert original_data[col].dtype == poisoned_data[col].dtype, (
                f"Data type for column '{col}' differs between datasets"
            )
    
    def test_no_invalid_species_introduced(self, poisoned_data):
        """Verify that no invalid species names were introduced during poisoning."""
        valid_species = {'setosa', 'versicolor', 'virginica'}
        actual_species = set(poisoned_data['species'].unique())
        
        invalid_species = actual_species - valid_species
        assert len(invalid_species) == 0, f"Invalid species introduced: {invalid_species}"
    
    def test_no_data_corruption(self, poisoned_data):
        """Verify that the poisoned dataset doesn't have data corruption."""
        # Check for NaN values
        assert not poisoned_data.isnull().any().any(), "Poisoned dataset contains NaN values"
        
        # Check for reasonable feature ranges (same as original validation)
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in feature_cols:
            assert (poisoned_data[col] >= 0).all(), f"Column '{col}' contains negative values"
            assert poisoned_data[col].dtype in ['float64', 'int64'], f"Column '{col}' has invalid data type"