import pandas as pd
import pytest
import os

@pytest.fixture(scope="module")
def iris_df():
    """Fixture to load the iris dataset once for all data validation tests."""
    data_path = 'data/iris.csv'
    if not os.path.exists(data_path):
        pytest.fail(f"Data file not found at {data_path}. Ensure the data file exists.")
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        pytest.fail(f"Failed to load data from {data_path}: {e}")

def test_data_file_exists():
    """Verify that the iris.csv data file exists."""
    assert os.path.exists('data/iris.csv'), "data/iris.csv not found."

def test_dataframe_shape(iris_df):
    """Verify the dataset has the expected number of rows and columns."""
    assert iris_df.shape == (150, 5), "Dataset should have 150 rows and 5 columns."

def test_column_names(iris_df):
    """Verify that all expected column names are present and correctly spelled."""
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert list(iris_df.columns) == expected_columns, "Column names do not match expected."

def test_no_missing_values(iris_df):
    """Verify that there are no missing values in the dataset."""
    assert not iris_df.isnull().any().any(), "Dataset contains missing values."

def test_feature_data_types(iris_df):
    """Verify that feature columns have numeric data types."""
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(iris_df[col]), f"Column '{col}' is not numeric."

def test_species_column_type(iris_df):
    """Verify that the 'species' column is of string/object type."""
    assert pd.api.types.is_string_dtype(iris_df['species']) or pd.api.types.is_object_dtype(iris_df['species']), \
        "Column 'species' is not of string/object type."

def test_unique_species_count(iris_df):
    """Verify the correct number and names of unique species."""
    expected_species = sorted(['setosa', 'versicolor', 'virginica'])
    actual_species = sorted(iris_df['species'].unique().tolist())
    assert actual_species == expected_species, "Species values do not match expected."
    assert len(iris_df['species'].unique()) == 3, "Expected 3 unique species."

def test_feature_value_ranges(iris_df):
    """Verify that feature values are non-negative."""
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in feature_cols:
        assert (iris_df[col] >= 0).all(), f"Column '{col}' contains negative values."