"""
Unit tests for evaluation/compare_models.py module.

This module tests model comparison functionality including metrics loading,
extraction, comparison, visualization, and report generation.
"""

import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from evaluation.compare_models import (
    setup_logger,
    load_metrics_from_json,
    extract_key_metrics,
    create_comparison_dataframe,
    plot_metrics_comparison,
    plot_radar_chart,
    generate_comparison_report
)


class TestSetupLogger:
    """Test cases for logger setup."""
    
    def test_setup_logger_default(self):
        """Test logger setup with default settings."""
        setup_logger()
        assert True
    
    def test_setup_logger_debug_mode(self):
        """Test logger setup in debug mode."""
        setup_logger(debug=True)
        assert True
    
    def test_setup_logger_info_mode(self):
        """Test logger setup in info mode."""
        setup_logger(debug=False)
        assert True


class TestLoadMetricsFromJSON:
    """Test cases for loading metrics from JSON files."""
    
    def test_load_metrics_valid_json(self):
        """Test loading valid JSON metrics file."""
        metrics_data = {
            "model": "RandomForest",
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics_data, f)
            metrics_path = Path(f.name)
        
        try:
            loaded = load_metrics_from_json(metrics_path)
            
            assert loaded == metrics_data
            assert loaded['model'] == 'RandomForest'
            assert loaded['metrics']['accuracy'] == 0.85
        finally:
            metrics_path.unlink()
    
    def test_load_metrics_nonexistent_file(self):
        """Test loading non-existent metrics file."""
        result = load_metrics_from_json(Path("nonexistent.json"))
        
        assert result == {}
    
    def test_load_metrics_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            metrics_path = Path(f.name)
        
        try:
            result = load_metrics_from_json(metrics_path)
            assert result == {}
        finally:
            metrics_path.unlink()
    
    def test_load_metrics_empty_file(self):
        """Test loading empty JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{}")
            metrics_path = Path(f.name)
        
        try:
            result = load_metrics_from_json(metrics_path)
            assert result == {}
        finally:
            metrics_path.unlink()
    
    def test_load_metrics_with_unicode(self):
        """Test loading metrics with unicode characters."""
        metrics_data = {
            "model": "Modelo con ñ",
            "metrics": {"accuracy": 0.85}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False)
            metrics_path = Path(f.name)
        
        try:
            loaded = load_metrics_from_json(metrics_path)
            assert "ñ" in loaded['model']
        finally:
            metrics_path.unlink()


class TestExtractKeyMetrics:
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
    
    def test_extract_key_metrics_empty_data(self):
        """Test extracting metrics from empty data."""
        key_metrics = extract_key_metrics({})
        
        assert key_metrics == {}
    
    def test_extract_key_metrics_missing_metrics_key(self):
        """Test extracting when 'metrics' key is missing."""
        metrics_data = {
            "model": "RandomForest",
            "params": {}
        }
        
        key_metrics = extract_key_metrics(metrics_data)
        
        assert key_metrics == {}
    
    def test_extract_key_metrics_partial_data(self):
        """Test extracting with partial metrics."""
        metrics_data = {
            "metrics": {
                "accuracy": 0.85
                # Missing other metrics
            }
        }
        
        key_metrics = extract_key_metrics(metrics_data)
        
        assert 'accuracy' in key_metrics
        assert 'precision' not in key_metrics
    
    def test_extract_key_metrics_all_metrics(self):
        """Test extracting all possible metrics."""
        metrics_data = {
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1": 0.85,
                "AUC_test_final": 0.92,
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
        
        assert len(key_metrics) == 8  # 5 direct + 3 weighted


class TestCreateComparisonDataFrame:
    """Test cases for creating comparison dataframes."""
    
    def test_create_comparison_dataframe_single_model(self):
        """Test creating dataframe with single model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir)
            
            metrics_data = {
                "model": "RandomForest",
                "metrics": {"accuracy": 0.85, "precision": 0.82}
            }
            
            with open(metrics_dir / "model1.json", 'w') as f:
                json.dump(metrics_data, f)
            
            df = create_comparison_dataframe(metrics_dir)
            
            assert len(df) == 1
            assert 'model_name' in df.columns
            assert 'accuracy' in df.columns
            assert df.iloc[0]['model_name'] == 'RandomForest'
    
    def test_create_comparison_dataframe_multiple_models(self):
        """Test creating dataframe with multiple models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir)
            
            models_data = [
                {"model": "RandomForest", "metrics": {"accuracy": 0.85}},
                {"model": "XGBoost", "metrics": {"accuracy": 0.87}},
                {"model": "LogisticRegression", "metrics": {"accuracy": 0.80}}
            ]
            
            for i, data in enumerate(models_data):
                with open(metrics_dir / f"model_{i}.json", 'w') as f:
                    json.dump(data, f)
            
            df = create_comparison_dataframe(metrics_dir)
            
            assert len(df) == 3
            assert 'RandomForest' in df['model_name'].values
            assert 'XGBoost' in df['model_name'].values
    
    def test_create_comparison_dataframe_empty_directory(self):
        """Test creating dataframe from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir)
            
            df = create_comparison_dataframe(metrics_dir)
            
            assert df.empty
    
    def test_create_comparison_dataframe_filters_invalid_files(self):
        """Test that invalid files are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir)
            
            # Valid file
            with open(metrics_dir / "valid.json", 'w') as f:
                json.dump({"model": "RF", "metrics": {"accuracy": 0.85}}, f)
            
            # Invalid file
            with open(metrics_dir / "invalid.json", 'w') as f:
                f.write("invalid json")
            
            df = create_comparison_dataframe(metrics_dir)
            
            assert len(df) == 1
    
    def test_create_comparison_dataframe_includes_model_file(self):
        """Test that model_file column is included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir)
            
            with open(metrics_dir / "test_model.json", 'w') as f:
                json.dump({"model": "RF", "metrics": {"accuracy": 0.85}}, f)
            
            df = create_comparison_dataframe(metrics_dir)
            
            assert 'model_file' in df.columns
            assert df.iloc[0]['model_file'] == 'test_model'


class TestPlotMetricsComparison:
    """Test cases for plotting metrics comparison."""
    
    def test_plot_metrics_comparison_basic(self):
        """Test basic metrics comparison plot."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B'],
            'accuracy': [0.85, 0.90],
            'precision': [0.83, 0.88]
        })
        
        metrics = ['accuracy', 'precision']
        
        plot_metrics_comparison(df, metrics)
        assert True
    
    def test_plot_metrics_comparison_with_save(self):
        """Test saving metrics comparison plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'model_name': ['Model A', 'Model B'],
                'accuracy': [0.85, 0.90]
            })
            
            save_path = Path(tmpdir) / "comparison.png"
            
            plot_metrics_comparison(df, ['accuracy'], save_path=save_path)
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_plot_metrics_comparison_missing_metrics(self):
        """Test plot with missing metrics."""
        df = pd.DataFrame({
            'model_name': ['Model A'],
            'accuracy': [0.85]
        })
        
        # Request metrics that don't exist
        metrics = ['accuracy', 'precision', 'recall']
        
        # Should handle gracefully
        plot_metrics_comparison(df, metrics)
        assert True
    
    def test_plot_metrics_comparison_no_available_metrics(self):
        """Test plot when no requested metrics are available."""
        df = pd.DataFrame({
            'model_name': ['Model A'],
            'accuracy': [0.85]
        })
        
        # Request metrics that don't exist
        metrics = ['precision', 'recall']
        
        # Should handle gracefully (log warning and return)
        plot_metrics_comparison(df, metrics)
        assert True
    
    def test_plot_metrics_comparison_custom_title(self):
        """Test plot with custom title."""
        df = pd.DataFrame({
            'model_name': ['Model A'],
            'accuracy': [0.85]
        })
        
        plot_metrics_comparison(df, ['accuracy'], title="Custom Title")
        assert True
    
    def test_plot_metrics_comparison_many_models(self):
        """Test plot with many models."""
        df = pd.DataFrame({
            'model_name': [f'Model {i}' for i in range(10)],
            'accuracy': np.random.rand(10)
        })
        
        plot_metrics_comparison(df, ['accuracy'])
        assert True


class TestPlotRadarChart:
    """Test cases for plotting radar charts."""
    
    def test_plot_radar_chart_basic(self):
        """Test basic radar chart plotting."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B'],
            'accuracy': [0.85, 0.90],
            'precision': [0.83, 0.88],
            'recall': [0.87, 0.92]
        })
        
        metrics = ['accuracy', 'precision', 'recall']
        
        plot_radar_chart(df, metrics)
        assert True
    
    def test_plot_radar_chart_with_save(self):
        """Test saving radar chart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'model_name': ['Model A', 'Model B'],
                'accuracy': [0.85, 0.90],
                'precision': [0.83, 0.88]
            })
            
            save_path = Path(tmpdir) / "radar.png"
            
            plot_radar_chart(df, ['accuracy', 'precision'], save_path=save_path)
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_plot_radar_chart_empty_dataframe(self):
        """Test radar chart with empty dataframe."""
        df = pd.DataFrame()
        
        # Should handle gracefully
        plot_radar_chart(df, ['accuracy'])
        assert True
    
    def test_plot_radar_chart_missing_metrics(self):
        """Test radar chart with missing metrics."""
        df = pd.DataFrame({
            'model_name': ['Model A'],
            'accuracy': [0.85]
        })
        
        metrics = ['accuracy', 'precision', 'recall']
        
        # Should handle gracefully
        plot_radar_chart(df, metrics)
        assert True
    
    def test_plot_radar_chart_single_model(self):
        """Test radar chart with single model."""
        df = pd.DataFrame({
            'model_name': ['Model A'],
            'accuracy': [0.85],
            'precision': [0.83],
            'recall': [0.87]
        })
        
        plot_radar_chart(df, ['accuracy', 'precision', 'recall'])
        assert True
    
    def test_plot_radar_chart_custom_title(self):
        """Test radar chart with custom title."""
        df = pd.DataFrame({
            'model_name': ['Model A'],
            'accuracy': [0.85],
            'precision': [0.83]
        })
        
        plot_radar_chart(df, ['accuracy', 'precision'], title="Custom Radar")
        assert True


class TestGenerateComparisonReport:
    """Test cases for generating comparison reports."""
    
    def test_generate_comparison_report_basic(self):
        """Test basic report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'model_name': ['Model A', 'Model B'],
                'accuracy': [0.85, 0.90],
                'precision': [0.83, 0.88]
            })
            
            output_path = Path(tmpdir) / "report.txt"
            
            generate_comparison_report(df, output_path)
            
            assert output_path.exists()
            
            # Verify content
            with open(output_path, 'r') as f:
                content = f.read()
            
            assert "REPORTE DE COMPARACIÓN DE MODELOS" in content
            assert "Model A" in content
            assert "Model B" in content
    
    def test_generate_comparison_report_includes_best_model(self):
        """Test that report includes best model per metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'model_name': ['Model A', 'Model B'],
                'accuracy': [0.85, 0.90],
                'precision': [0.88, 0.83]
            })
            
            output_path = Path(tmpdir) / "report.txt"
            
            generate_comparison_report(df, output_path)
            
            with open(output_path, 'r') as f:
                content = f.read()
            
            assert "Mejor Modelo por Métrica" in content
            assert "accuracy" in content
            assert "precision" in content
    
    def test_generate_comparison_report_includes_ranking(self):
        """Test that report includes overall ranking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'model_name': ['Model A', 'Model B', 'Model C'],
                'accuracy': [0.85, 0.90, 0.82],
                'precision': [0.83, 0.88, 0.80]
            })
            
            output_path = Path(tmpdir) / "report.txt"
            
            generate_comparison_report(df, output_path)
            
            with open(output_path, 'r') as f:
                content = f.read()
            
            assert "Ranking General" in content
            assert "1." in content
            assert "2." in content
            assert "3." in content
    
    def test_generate_comparison_report_single_model(self):
        """Test report generation with single model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'model_name': ['Model A'],
                'accuracy': [0.85]
            })
            
            output_path = Path(tmpdir) / "report.txt"
            
            generate_comparison_report(df, output_path)
            
            assert output_path.exists()
    
    def test_generate_comparison_report_with_unicode(self):
        """Test report generation with unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'model_name': ['Modelo con ñ'],
                'accuracy': [0.85]
            })
            
            output_path = Path(tmpdir) / "report.txt"
            
            generate_comparison_report(df, output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "ñ" in content


class TestIntegration:
    """Integration tests for complete comparison workflow."""
    
    def test_complete_comparison_workflow(self):
        """Test complete workflow: load → extract → compare → visualize → report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dir = Path(tmpdir) / "metrics"
            output_dir = Path(tmpdir) / "output"
            metrics_dir.mkdir()
            output_dir.mkdir()
            
            # Create multiple model metrics
            models = [
                {"model": "RandomForest", "metrics": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88}},
                {"model": "XGBoost", "metrics": {"accuracy": 0.87, "precision": 0.84, "recall": 0.90}},
                {"model": "LogisticRegression", "metrics": {"accuracy": 0.80, "precision": 0.78, "recall": 0.82}}
            ]
            
            for i, model_data in enumerate(models):
                with open(metrics_dir / f"model_{i}.json", 'w') as f:
                    json.dump(model_data, f)
            
            # Create comparison dataframe
            df = create_comparison_dataframe(metrics_dir)
            
            assert len(df) == 3
            
            # Generate visualizations
            plot_metrics_comparison(
                df,
                ['accuracy', 'precision', 'recall'],
                save_path=output_dir / "comparison.png"
            )
            
            plot_radar_chart(
                df,
                ['accuracy', 'precision', 'recall'],
                save_path=output_dir / "radar.png"
            )
            
            # Generate report
            generate_comparison_report(df, output_dir / "report.txt")
            
            # Verify all outputs
            assert (output_dir / "comparison.png").exists()
            assert (output_dir / "radar.png").exists()
            assert (output_dir / "report.txt").exists()
    
    def test_identify_best_model(self):
        """Test identifying the best model."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B', 'Model C'],
            'accuracy': [0.85, 0.90, 0.82],
            'precision': [0.83, 0.88, 0.80],
            'recall': [0.87, 0.92, 0.85]
        })
        
        # Calculate average score
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df['avg_score'] = df[numeric_cols].mean(axis=1)
        
        best_model_idx = df['avg_score'].idxmax()
        best_model = df.loc[best_model_idx, 'model_name']
        
        assert best_model == 'Model B'


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_dataframe_with_nan_values(self):
        """Test handling dataframe with NaN values."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B'],
            'accuracy': [0.85, np.nan],
            'precision': [np.nan, 0.88]
        })
        
        # Should handle gracefully
        plot_metrics_comparison(df, ['accuracy', 'precision'])
        assert True
    
    def test_empty_metrics_list(self):
        """Test with empty metrics list."""
        df = pd.DataFrame({
            'model_name': ['Model A'],
            'accuracy': [0.85]
        })
        
        plot_metrics_comparison(df, [])
        assert True
    
    def test_dataframe_with_no_numeric_columns(self):
        """Test dataframe with no numeric columns."""
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B'],
            'description': ['Desc A', 'Desc B']
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            
            # Should handle gracefully
            generate_comparison_report(df, output_path)
            assert output_path.exists()


class TestPlotCleanup:
    """Test that plots are properly cleaned up."""
    
    def test_plots_close_properly(self):
        """Test that matplotlib figures are closed after plotting."""
        initial_figs = len(plt.get_fignums())
        
        df = pd.DataFrame({
            'model_name': ['Model A', 'Model B'],
            'accuracy': [0.85, 0.90]
        })
        
        plot_metrics_comparison(df, ['accuracy'])
        plot_radar_chart(df, ['accuracy'])
        
        final_figs = len(plt.get_fignums())
        
        # Should not leave figures open
        assert final_figs == initial_figs
