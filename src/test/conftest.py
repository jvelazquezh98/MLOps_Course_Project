"""
Pytest configuration and fixtures for the MLOps Course Project tests.

This module contains shared fixtures and configuration for all tests
in the MLOps Course Project.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pickle


@pytest.fixture
def temp_models_dir():
    """Create a temporary models directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = Path(temp_dir) / "models"
        models_dir.mkdir()
        yield models_dir


@pytest.fixture
def sample_csv_data():
    """Provide sample CSV data for testing."""
    return "feature1,feature2,feature3\n1,2,3\n4,5,6\n7,8,9"


@pytest.fixture
def sample_dataframe():
    """Provide sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1, 4, 7],
        'feature2': [2, 5, 8],
        'feature3': [3, 6, 9]
    })


@pytest.fixture
def mock_model():
    """Create a mock ML model for testing."""
    model = MagicMock()
    model.predict.return_value = [0.8, 0.2, 0.9]
    return model


@pytest.fixture
def sample_metadata():
    """Provide sample metadata for testing."""
    return {
        "timestamp": "2025-10-10T15:33:42.103451",
        "data_engineer_activities": {
            "outlier_treatment": True,
            "missing_value_imputation": True,
            "categorical_encoding": True,
            "feature_engineering": True,
            "pipeline_creation": True,
            "dimensionality_reduction": True
        },
        "dataset_info": {
            "original_shape": [40436, 62],
            "engineered_shape": [40436, 68],
            "processed_shape": [40436, 34],
            "new_features_created": [
                "content_complexity",
                "media_ratio",
                "self_reference_ratio",
                "title_engagement"
            ]
        }
    }


@pytest.fixture
def create_test_model_file(temp_models_dir, mock_model):
    """Create a test model file in the temporary directory."""
    def _create_model(model_name: str):
        model_path = temp_models_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)
        return model_path
    
    return _create_model


@pytest.fixture
def create_test_csv_file():
    """Create a test CSV file."""
    def _create_csv(content: str, filename: str = "test.csv"):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            f.flush()
            return f.name
    
    return _create_csv


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )


# Custom pytest markers for test categorization
pytestmark = [
    pytest.mark.asyncio,
]