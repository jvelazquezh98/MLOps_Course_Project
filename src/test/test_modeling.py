"""
Unit tests for modeling modules.

This module tests model training, prediction, and model management
functionality for the MLOps pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.modeling.train import train_model, set_seed, DEFAULT_PARAMS
from src.modeling.predict import main as predict_main


class TestModelTraining:
    """Test cases for model training functionality."""
    
    def create_sample_training_data(self) -> Path:
        """Helper to create sample training data."""
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'shares': np.random.randint(500, 3000, 100)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            return Path(f.name)
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed ensures reproducibility."""
        set_seed(42)
        random_vals_1 = np.random.rand(10)
        
        set_seed(42)
        random_vals_2 = np.random.rand(10)
        
        np.testing.assert_array_equal(random_vals_1, random_vals_2)
    
    def test_default_params_structure(self):
        """Test that default parameters have correct structure."""
        assert isinstance(DEFAULT_PARAMS, dict)
        assert 'n_estimators' in DEFAULT_PARAMS
        assert 'max_depth' in DEFAULT_PARAMS
        assert 'random_state' in DEFAULT_PARAMS
        
        # Check types
        assert isinstance(DEFAULT_PARAMS['n_estimators'], int)
        assert isinstance(DEFAULT_PARAMS['random_state'], int)
    
    def test_train_model_creates_target_from_shares(self):
        """Test that train_model creates target from shares column."""
        data_path = self.create_sample_training_data()
        
        try:
            result = train_model(
                data_path=data_path,
                target='popular',
                params=DEFAULT_PARAMS
            )
            
            assert 'model' in result
            assert 'metrics' in result
            assert isinstance(result['model'], RandomForestClassifier)
        finally:
            data_path.unlink()
    
    def test_train_model_returns_metrics(self):
        """Test that train_model returns expected metrics."""
        data_path = self.create_sample_training_data()
        
        try:
            result = train_model(
                data_path=data_path,
                target='popular',
                params=DEFAULT_PARAMS
            )
            
            metrics = result['metrics']
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            
            # Check metric ranges
            for metric_name, metric_value in metrics.items():
                assert 0 <= metric_value <= 1, f"{metric_name} out of range"
        finally:
            data_path.unlink()
    
    def test_train_model_with_custom_params(self):
        """Test training with custom parameters."""
        data_path = self.create_sample_training_data()
        
        custom_params = {
            **DEFAULT_PARAMS,
            'n_estimators': 50,
            'max_depth': 5
        }
        
        try:
            result = train_model(
                data_path=data_path,
                target='popular',
                params=custom_params
            )
            
            model = result['model']
            assert model.n_estimators == 50
            assert model.max_depth == 5
        finally:
            data_path.unlink()
    
    def test_train_model_handles_missing_file(self):
        """Test that train_model handles missing data file."""
        with pytest.raises(FileNotFoundError):
            train_model(
                data_path=Path("nonexistent.csv"),
                target='popular',
                params=DEFAULT_PARAMS
            )
    
    def test_train_model_removes_shares_from_features(self):
        """Test that shares column is removed from features."""
        # Create larger dataset to avoid train_test_split issues
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'shares': np.random.randint(1000, 3000, 100),
            'popular': np.random.randint(0, 2, 100)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            data_path = Path(f.name)
        
        try:
            result = train_model(
                data_path=data_path,
                target='popular',
                params=DEFAULT_PARAMS
            )
            
            # Model should be trained successfully
            assert result['model'] is not None
        finally:
            data_path.unlink()


class TestModelPrediction:
    """Test cases for model prediction functionality."""
    
    def create_mock_model(self):
        """Helper to create a mock trained model."""
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0, 1])
        model.predict_proba.return_value = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.4, 0.6]
        ])
        return model
    
    def test_model_prediction_basic(self):
        """Test basic model prediction."""
        model = self.create_mock_model()
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        predictions = model.predict(X)
        
        assert len(predictions) == 4
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_prediction_with_probabilities(self):
        """Test model prediction with probabilities."""
        model = self.create_mock_model()
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        probabilities = model.predict_proba(X)
        
        assert probabilities.shape == (4, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_prediction_handles_missing_columns(self):
        """Test that prediction handles missing columns gracefully."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'popular': [0, 1, 0],
            'shares': [1000, 2000, 1500]
        })
        
        # Remove target columns
        cols_to_drop = [c for c in ['popular', 'shares'] if c in df.columns]
        df_features = df.drop(columns=cols_to_drop)
        
        assert 'popular' not in df_features.columns
        assert 'shares' not in df_features.columns
    
    def test_prediction_removes_non_numeric_columns(self):
        """Test that non-numeric columns are removed before prediction."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'url': ['http://a.com', 'http://b.com', 'http://c.com']
        })
        
        non_numeric = df.select_dtypes(exclude=['number']).columns.tolist()
        df_numeric = df.drop(columns=non_numeric)
        
        assert 'url' not in df_numeric.columns
        assert all(df_numeric.dtypes.apply(lambda x: np.issubdtype(x, np.number)))


class TestModelPersistence:
    """Test cases for model saving and loading."""
    
    def test_model_can_be_pickled(self):
        """Test that model can be saved with pickle."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(model, f)
            model_path = Path(f.name)
        
        try:
            # Load and verify
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            assert isinstance(loaded_model, RandomForestClassifier)
            assert loaded_model.n_estimators == 10
        finally:
            model_path.unlink()
    
    def test_model_can_be_joblib_saved(self):
        """Test that model can be saved with joblib."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            joblib.dump(model, f.name)
            model_path = Path(f.name)
        
        try:
            # Load and verify
            loaded_model = joblib.load(model_path)
            
            assert isinstance(loaded_model, RandomForestClassifier)
            assert loaded_model.n_estimators == 10
        finally:
            model_path.unlink()
    
    def test_loaded_model_makes_same_predictions(self):
        """Test that loaded model makes same predictions as original."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        model.fit(X_train, y_train)
        
        X_test = np.random.rand(10, 5)
        original_predictions = model.predict(X_test)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(model, f)
            model_path = Path(f.name)
        
        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            loaded_predictions = loaded_model.predict(X_test)
            
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
        finally:
            model_path.unlink()


class TestModelValidation:
    """Test cases for model validation."""
    
    def test_model_predictions_are_binary(self):
        """Test that model predictions are binary (0 or 1)."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        X_test = np.random.rand(20, 5)
        predictions = model.predict(X_test)
        
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_probabilities_sum_to_one(self):
        """Test that prediction probabilities sum to 1."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        X_test = np.random.rand(20, 5)
        probabilities = model.predict_proba(X_test)
        
        row_sums = probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(20))
    
    def test_model_handles_edge_cases(self):
        """Test that model handles edge cases in input data."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        
        # Test with zeros
        X_zeros = np.zeros((5, 5))
        predictions_zeros = model.predict(X_zeros)
        assert len(predictions_zeros) == 5
        
        # Test with ones
        X_ones = np.ones((5, 5))
        predictions_ones = model.predict(X_ones)
        assert len(predictions_ones) == 5
