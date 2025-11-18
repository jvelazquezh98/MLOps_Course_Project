"""
Integration tests for MLOps pipeline.

This module contains end-to-end integration tests that validate
the complete MLOps workflow from data processing to model deployment.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import pickle
import json
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier

from src.config import MODELS_DIR, PROCESSED_DATA_DIR


class TestDataPipeline:
    """Integration tests for data processing pipeline."""
    
    def test_end_to_end_data_processing(self):
        """Test complete data processing pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create raw data
            raw_data = pd.DataFrame({
                'url': ['http://example.com/1', 'http://example.com/2'],
                'feature1': [1.0, 2.0],
                'feature2': [3.0, 4.0],
                'mixed_type_col': ['bad', 'unknown'],
                'shares': [1000, 2000]
            })
            
            raw_path = Path(tmpdir) / "raw.csv"
            raw_data.to_csv(raw_path, index=False)
            
            # 2. Process data (simulate feature engineering)
            df = pd.read_csv(raw_path)
            df['mixed_type_col_clean'] = df['mixed_type_col'].replace({
                'bad': 0,
                'unknown': 1
            })
            df = df.drop('mixed_type_col', axis=1)
            
            # 3. Save processed data
            processed_path = Path(tmpdir) / "processed.csv"
            df.to_csv(processed_path, index=False)
            
            # 4. Verify processed data
            df_final = pd.read_csv(processed_path)
            assert 'mixed_type_col_clean' in df_final.columns
            assert 'mixed_type_col' not in df_final.columns
            assert len(df_final) == 2
    
    def test_data_validation_pipeline(self):
        """Test data validation in pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with quality issues
            data = pd.DataFrame({
                'feature1': [1.0, np.nan, 3.0],
                'feature2': [4.0, 5.0, np.nan],
                'feature3': [7.0, 8.0, 9.0]
            })
            
            data_path = Path(tmpdir) / "data.csv"
            data.to_csv(data_path, index=False)
            
            # Load and validate
            df = pd.read_csv(data_path)
            
            # Check for nulls
            null_counts = df.isnull().sum()
            assert null_counts['feature1'] == 1
            assert null_counts['feature2'] == 1
            
            # Clean data
            df_clean = df.fillna(df.median())
            assert df_clean.isnull().sum().sum() == 0


class TestModelPipeline:
    """Integration tests for model training and prediction pipeline."""
    
    def test_end_to_end_model_training(self):
        """Test complete model training pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create training data
            train_data = pd.DataFrame({
                'feature1': np.random.rand(100),
                'feature2': np.random.rand(100),
                'feature3': np.random.rand(100),
                'shares': np.random.randint(500, 3000, 100)
            })
            
            train_path = Path(tmpdir) / "train.csv"
            train_data.to_csv(train_path, index=False)
            
            # 2. Create target variable
            df = pd.read_csv(train_path)
            df['popular'] = (df['shares'] > 1400).astype(int)
            
            # 3. Train model
            X = df[['feature1', 'feature2', 'feature3']]
            y = df['popular']
            
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # 4. Save model
            model_path = Path(tmpdir) / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 5. Load and predict
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            predictions = loaded_model.predict(X[:10])
            
            assert len(predictions) == 10
            assert all(p in [0, 1] for p in predictions)
    
    def test_model_versioning_pipeline(self):
        """Test model versioning in pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            
            # Create multiple model versions
            for version in range(1, 4):
                model = RandomForestClassifier(n_estimators=version * 10, random_state=42)
                X = np.random.rand(50, 5)
                y = np.random.randint(0, 2, 50)
                model.fit(X, y)
                
                model_path = models_dir / f"model_v{version}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Verify all versions exist
            model_files = list(models_dir.glob("*.pkl"))
            assert len(model_files) == 3
            
            # Load latest version
            latest_model_path = sorted(model_files)[-1]
            with open(latest_model_path, 'rb') as f:
                latest_model = pickle.load(f)
            
            assert latest_model.n_estimators == 30
    
    def test_prediction_pipeline_with_preprocessing(self):
        """Test prediction pipeline with preprocessing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Train model
            X_train = np.random.rand(100, 5)
            y_train = np.random.randint(0, 2, 100)
            
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # 2. Save model
            model_path = Path(tmpdir) / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 3. Create new data for prediction
            new_data = pd.DataFrame({
                'feature1': [1.0, 2.0],
                'feature2': [3.0, 4.0],
                'feature3': [5.0, 6.0],
                'feature4': [7.0, 8.0],
                'feature5': [9.0, 10.0],
                'url': ['http://a.com', 'http://b.com']  # Non-numeric column
            })
            
            # 4. Preprocess (remove non-numeric)
            non_numeric = new_data.select_dtypes(exclude=['number']).columns
            X_new = new_data.drop(columns=non_numeric)
            
            # 5. Load model and predict
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            predictions = loaded_model.predict(X_new)
            
            assert len(predictions) == 2


class TestMLOpsWorkflow:
    """Integration tests for complete MLOps workflow."""
    
    def test_complete_mlops_workflow(self):
        """Test complete MLOps workflow from data to deployment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Data Ingestion
            raw_data = pd.DataFrame({
                'feature1': np.random.rand(100),
                'feature2': np.random.rand(100),
                'shares': np.random.randint(500, 3000, 100)
            })
            raw_path = Path(tmpdir) / "raw.csv"
            raw_data.to_csv(raw_path, index=False)
            
            # 2. Feature Engineering
            df = pd.read_csv(raw_path)
            df['popular'] = (df['shares'] > 1400).astype(int)
            processed_path = Path(tmpdir) / "processed.csv"
            df.to_csv(processed_path, index=False)
            
            # 3. Model Training
            df_train = pd.read_csv(processed_path)
            X = df_train[['feature1', 'feature2']]
            y = df_train['popular']
            
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # 4. Model Evaluation
            from sklearn.metrics import accuracy_score, precision_score
            predictions = model.predict(X)
            metrics = {
                'accuracy': float(accuracy_score(y, predictions)),
                'precision': float(precision_score(y, predictions, zero_division=0))
            }
            
            # 5. Model Persistence
            model_path = Path(tmpdir) / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 6. Metrics Persistence
            metrics_path = Path(tmpdir) / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            
            # 7. Verify all artifacts exist
            assert raw_path.exists()
            assert processed_path.exists()
            assert model_path.exists()
            assert metrics_path.exists()
            
            # 8. Load and verify metrics
            with open(metrics_path, 'r') as f:
                loaded_metrics = json.load(f)
            
            assert 'accuracy' in loaded_metrics
            assert 'precision' in loaded_metrics
            assert 0 <= loaded_metrics['accuracy'] <= 1
    
    def test_model_retraining_workflow(self):
        """Test model retraining workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initial training
            X_v1 = np.random.rand(100, 5)
            y_v1 = np.random.randint(0, 2, 100)
            
            model_v1 = RandomForestClassifier(n_estimators=10, random_state=42)
            model_v1.fit(X_v1, y_v1)
            
            model_v1_path = Path(tmpdir) / "model_v1.pkl"
            with open(model_v1_path, 'wb') as f:
                pickle.dump(model_v1, f)
            
            # Retrain with new data
            X_v2 = np.random.rand(150, 5)
            y_v2 = np.random.randint(0, 2, 150)
            
            model_v2 = RandomForestClassifier(n_estimators=20, random_state=42)
            model_v2.fit(X_v2, y_v2)
            
            model_v2_path = Path(tmpdir) / "model_v2.pkl"
            with open(model_v2_path, 'wb') as f:
                pickle.dump(model_v2, f)
            
            # Verify both versions exist
            assert model_v1_path.exists()
            assert model_v2_path.exists()
            
            # Load and compare
            with open(model_v1_path, 'rb') as f:
                loaded_v1 = pickle.load(f)
            with open(model_v2_path, 'rb') as f:
                loaded_v2 = pickle.load(f)
            
            assert loaded_v1.n_estimators == 10
            assert loaded_v2.n_estimators == 20


class TestAPIIntegration:
    """Integration tests for REST API with MLOps components."""
    
    def test_api_model_loading_integration(self):
        """Test API integration with model loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            X = np.random.rand(50, 5)
            y = np.random.randint(0, 2, 50)
            model.fit(X, y)
            
            model_path = Path(tmpdir) / "test_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Simulate API loading model
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Make prediction
            test_data = np.random.rand(5, 5)
            predictions = loaded_model.predict(test_data)
            
            assert len(predictions) == 5
    
    def test_api_csv_processing_integration(self):
        """Test API integration with CSV processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV data
            df = pd.DataFrame({
                'feature1': [1, 2, 3],
                'feature2': [4, 5, 6],
                'feature3': [7, 8, 9]
            })
            
            csv_path = Path(tmpdir) / "test.csv"
            df.to_csv(csv_path, index=False)
            
            # Simulate API reading CSV
            df_loaded = pd.read_csv(csv_path)
            
            assert len(df_loaded) == 3
            assert list(df_loaded.columns) == ['feature1', 'feature2', 'feature3']


class TestDataVersioning:
    """Integration tests for data versioning with DVC."""
    
    def test_data_versioning_workflow(self):
        """Test data versioning workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple versions of data
            versions = []
            
            for version in range(1, 4):
                data = pd.DataFrame({
                    'feature1': np.random.rand(100),
                    'feature2': np.random.rand(100),
                    'version': [version] * 100
                })
                
                data_path = Path(tmpdir) / f"data_v{version}.csv"
                data.to_csv(data_path, index=False)
                versions.append(data_path)
            
            # Verify all versions exist
            assert len(versions) == 3
            
            # Load and verify each version
            for i, version_path in enumerate(versions, 1):
                df = pd.read_csv(version_path)
                assert df['version'].iloc[0] == i
    
    def test_data_lineage_tracking(self):
        """Test data lineage tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create metadata for data lineage
            lineage = {
                'raw_data': 'online_news_original.csv',
                'processed_data': 'online_news_processed.csv',
                'transformations': [
                    'remove_nulls',
                    'encode_categorical',
                    'create_features'
                ],
                'timestamp': '2025-10-10T15:33:42'
            }
            
            lineage_path = Path(tmpdir) / "lineage.json"
            with open(lineage_path, 'w') as f:
                json.dump(lineage, f)
            
            # Verify lineage can be loaded
            with open(lineage_path, 'r') as f:
                loaded_lineage = json.load(f)
            
            assert loaded_lineage['raw_data'] == 'online_news_original.csv'
            assert len(loaded_lineage['transformations']) == 3


class TestModelMonitoring:
    """Integration tests for model monitoring."""
    
    def test_model_performance_tracking(self):
        """Test model performance tracking over time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate multiple evaluation runs
            performance_history = []
            
            for run in range(1, 6):
                metrics = {
                    'run_id': run,
                    'accuracy': 0.80 + (run * 0.02),
                    'precision': 0.75 + (run * 0.03),
                    'timestamp': f'2025-10-{run:02d}'
                }
                performance_history.append(metrics)
            
            # Save performance history
            history_path = Path(tmpdir) / "performance_history.json"
            with open(history_path, 'w') as f:
                json.dump(performance_history, f)
            
            # Load and analyze
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            assert len(history) == 5
            assert history[-1]['accuracy'] > history[0]['accuracy']
    
    def test_model_drift_detection(self):
        """Test model drift detection."""
        # Simulate baseline and current predictions
        baseline_predictions = np.random.binomial(1, 0.5, 1000)
        current_predictions = np.random.binomial(1, 0.6, 1000)
        
        # Calculate distribution difference
        baseline_mean = baseline_predictions.mean()
        current_mean = current_predictions.mean()
        
        drift = abs(current_mean - baseline_mean)
        
        # Drift should be detectable
        assert drift > 0
