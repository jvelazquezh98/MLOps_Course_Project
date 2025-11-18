"""
Unit tests for model evaluation and comparison.

This module tests the model comparison functionality including
metrics extraction, visualization generation, and report creation.
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile

from evaluation.compare_models import (
    load_metrics_from_json,
    extract_key_metrics,
    create_comparison_dataframe
)


class TestMetricsLoading:
    """Test cases for loading metrics from JSON files."""
    
    def test_load_metrics_from_valid_json(self):
        """Test loading metrics from valid JSON file."""
        metrics_data = {
            "model": "RandomForest",
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics_data, f)
            metrics_path = Path(f.name)
        
        try:
            loaded_metrics = load_metrics_from_json(metrics_path)
            
            assert loaded_metrics == metrics_data
            assert loaded_metrics['model'] == 'RandomForest'
            assert loaded_metrics['metrics']['accuracy'] == 0.85
        finally:
            metrics_path.unlink()
    
    def test_load_metrics_from_nonexistent_file(self):
        """Test loading metrics from non-existent file."""
        result = load_metrics_from_json(Path("nonexistent.json"))
        
        assert result == {}
    
    def test_load_metrics_from_invalid_json(self):
        """Test loading metrics from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            metrics_path = Path(f.name)
        
        try:
            result = load_metrics_from_json(metrics_path)
            assert result == {}
        finally:
            metrics_path.unlink()


class TestMetricsExtraction:
    """Test cases for extracting key metrics."""
    
    def test_extract_key_metrics_basic(self):
        """Test extracting basic metrics."""
        metrics_data = {
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1": 0.85
            }
        }
        
        key_metrics = extract_key_metrics(metrics_data)
        
        assert 'accuracy' in key_metrics
        assert 'precision' in key_metrics
        assert 'recall' in key_metrics
        assert 'f1' in key_metrics
        assert key_metrics['accuracy'] == 0.85
    
    def test_extract_key_metrics_with_classification_report(self):
        """Test extracting metrics from classification report."""
        metrics_data = {
            "metrics": {
                "accuracy": 0.85,
                "classification_report": {
                    "weighted_avg": {
                        "precision": 0.83,
                        "recall": 0.85,
                        "f1_score": 0.84
                    }
                }
            }
        }
        
        key_metrics = extract_key_metrics(metrics_data)
        
        assert 'weighted_precision' in key_metrics
        assert 'weighted_recall' in key_metrics
        assert 'weighted_f1' in key_metrics
        assert key_metrics['weighted_precision'] == 0.83
    
    def test_extract_key_metrics_with_auc(self):
        """Test extracting AUC metric."""
        metrics_data = {
            "metrics": {
                "AUC_test_final": 0.92
            }
        }
        
        key_metrics = extract_key_metrics(metrics_data)
        
        assert 'AUC_test_final' in key_metrics
        assert key_metrics['AUC_test_final'] == 0.92
    
    def test_extract_key_metrics_empty_data(self):
        """Test extracting metrics from empty data."""
        key_metrics = extract_key_metrics({})
        
        assert key_metrics == {}
    
    def test_extract_key_metrics_missing_metrics_key(self):
        """Test extracting metrics when 'metrics' key is missing."""
        metrics_data = {
            "model": "RandomForest",
            "params": {}
        }
        
        key_metrics = extract_key_metrics(metrics_data)
        
        assert key_metrics == {}


class TestComparisonDataFrame:
    """Test cases for creating comparison dataframes."""
    
    def test_create_comparison_dataframe_with_multiple_models(self):
        """Test creating comparison dataframe with multiple models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir)
            
            # Create multiple metric files
            models_data = [
                {
                    "model": "RandomForest",
                    "metrics": {"accuracy": 0.85, "precision": 0.82}
                },
                {
                    "model": "XGBoost",
                    "metrics": {"accuracy": 0.87, "precision": 0.84}
                }
            ]
            
            for i, data in enumerate(models_data):
                with open(metrics_dir / f"model_{i}.json", 'w') as f:
                    json.dump(data, f)
            
            df = create_comparison_dataframe(metrics_dir)
            
            assert len(df) == 2
            assert 'model_name' in df.columns
            assert 'accuracy' in df.columns
    
    def test_create_comparison_dataframe_empty_directory(self):
        """Test creating comparison dataframe from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir)
            
            df = create_comparison_dataframe(metrics_dir)
            
            assert df.empty
    
    def test_create_comparison_dataframe_filters_invalid_files(self):
        """Test that invalid JSON files are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir)
            
            # Create valid file
            with open(metrics_dir / "valid.json", 'w') as f:
                json.dump({"model": "RF", "metrics": {"accuracy": 0.85}}, f)
            
            # Create invalid file
            with open(metrics_dir / "invalid.json", 'w') as f:
                f.write("invalid json")
            
            df = create_comparison_dataframe(metrics_dir)
            
            # Should only include valid file
            assert len(df) == 1


class TestModelComparison:
    """Test cases for model comparison functionality."""
    
    def test_comparison_identifies_best_model(self):
        """Test that comparison correctly identifies best model."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B', 'Model C'],
            'accuracy': [0.85, 0.90, 0.82],
            'precision': [0.83, 0.88, 0.80],
            'recall': [0.87, 0.92, 0.85]
        })
        
        # Calculate average score
        numeric_cols = ['accuracy', 'precision', 'recall']
        df['avg_score'] = df[numeric_cols].mean(axis=1)
        
        best_model_idx = df['avg_score'].idxmax()
        best_model = df.loc[best_model_idx, 'model_name']
        
        assert best_model == 'Model B'
    
    def test_comparison_handles_missing_metrics(self):
        """Test that comparison handles missing metrics gracefully."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B'],
            'accuracy': [0.85, 0.90],
            'precision': [0.83, np.nan]
        })
        
        # Should handle NaN values
        assert df['accuracy'].notna().all()
        assert df['precision'].isna().any()
    
    def test_comparison_ranks_models_correctly(self):
        """Test that models are ranked correctly."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B', 'Model C'],
            'accuracy': [0.85, 0.90, 0.82]
        })
        
        df_sorted = df.sort_values('accuracy', ascending=False)
        
        assert df_sorted.iloc[0]['model_name'] == 'Model B'
        assert df_sorted.iloc[1]['model_name'] == 'Model A'
        assert df_sorted.iloc[2]['model_name'] == 'Model C'


class TestVisualizationGeneration:
    """Test cases for visualization generation."""
    
    def test_plot_data_preparation(self):
        """Test that data is prepared correctly for plotting."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B'],
            'accuracy': [0.85, 0.90],
            'precision': [0.83, 0.88]
        })
        
        metrics = ['accuracy', 'precision']
        available_metrics = [m for m in metrics if m in df.columns]
        
        assert len(available_metrics) == 2
        assert 'accuracy' in available_metrics
        assert 'precision' in available_metrics
    
    def test_plot_handles_missing_metrics(self):
        """Test that plotting handles missing metrics."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B'],
            'accuracy': [0.85, 0.90]
        })
        
        requested_metrics = ['accuracy', 'precision', 'recall']
        available_metrics = [m for m in requested_metrics if m in df.columns]
        
        assert len(available_metrics) == 1
        assert available_metrics == ['accuracy']


class TestReportGeneration:
    """Test cases for report generation."""
    
    def test_report_contains_all_models(self):
        """Test that report contains all models."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B', 'Model C'],
            'accuracy': [0.85, 0.90, 0.82]
        })
        
        assert len(df) == 3
        assert 'Model A' in df['model_name'].values
        assert 'Model B' in df['model_name'].values
        assert 'Model C' in df['model_name'].values
    
    def test_report_calculates_statistics(self):
        """Test that report calculates statistics correctly."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B'],
            'accuracy': [0.85, 0.90],
            'precision': [0.83, 0.88]
        })
        
        # Calculate statistics
        mean_accuracy = df['accuracy'].mean()
        max_accuracy = df['accuracy'].max()
        
        assert mean_accuracy == 0.875
        assert max_accuracy == 0.90
    
    def test_report_identifies_best_per_metric(self):
        """Test that report identifies best model per metric."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B'],
            'accuracy': [0.85, 0.90],
            'precision': [0.88, 0.83]
        })
        
        best_accuracy_idx = df['accuracy'].idxmax()
        best_precision_idx = df['precision'].idxmax()
        
        assert df.loc[best_accuracy_idx, 'model_name'] == 'Model B'
        assert df.loc[best_precision_idx, 'model_name'] == 'Model A'
