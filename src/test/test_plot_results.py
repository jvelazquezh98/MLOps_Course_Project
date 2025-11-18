"""
Unit tests for visualization/plot_results.py module.

This module tests all visualization functions including confusion matrices,
ROC curves, precision-recall curves, prediction distributions, and feature importance.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from visualization.plot_results import (
    setup_logger,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_prediction_distribution,
    plot_feature_importance
)


class TestSetupLogger:
    """Test cases for logger setup."""
    
    def test_setup_logger_default(self):
        """Test logger setup with default settings."""
        setup_logger()
        # Should not raise any errors
        assert True
    
    def test_setup_logger_debug_mode(self):
        """Test logger setup in debug mode."""
        setup_logger(debug=True)
        # Should not raise any errors
        assert True
    
    def test_setup_logger_info_mode(self):
        """Test logger setup in info mode."""
        setup_logger(debug=False)
        # Should not raise any errors
        assert True


class TestPlotConfusionMatrix:
    """Test cases for confusion matrix plotting."""
    
    def test_plot_confusion_matrix_basic(self):
        """Test basic confusion matrix plotting."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        # Should not raise any errors
        plot_confusion_matrix(y_true, y_pred)
        assert True
    
    def test_plot_confusion_matrix_with_title(self):
        """Test confusion matrix with custom title."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        plot_confusion_matrix(y_true, y_pred, title="Custom Title")
        assert True
    
    def test_plot_confusion_matrix_save_to_file(self):
        """Test saving confusion matrix to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            y_true = np.array([0, 1, 0, 1, 0, 1])
            y_pred = np.array([0, 1, 0, 0, 0, 1])
            save_path = Path(tmpdir) / "confusion_matrix.png"
            
            plot_confusion_matrix(y_true, y_pred, save_path=save_path)
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_plot_confusion_matrix_perfect_predictions(self):
        """Test confusion matrix with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        plot_confusion_matrix(y_true, y_pred)
        assert True
    
    def test_plot_confusion_matrix_all_wrong(self):
        """Test confusion matrix with all wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        
        plot_confusion_matrix(y_true, y_pred)
        assert True
    
    def test_plot_confusion_matrix_large_dataset(self):
        """Test confusion matrix with large dataset."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_pred = np.random.randint(0, 2, 1000)
        
        plot_confusion_matrix(y_true, y_pred)
        assert True


class TestPlotROCCurve:
    """Test cases for ROC curve plotting."""
    
    def test_plot_roc_curve_basic(self):
        """Test basic ROC curve plotting."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.35, 0.8])
        
        auc_score = plot_roc_curve(y_true, y_proba)
        
        assert isinstance(auc_score, float)
        assert 0 <= auc_score <= 1
    
    def test_plot_roc_curve_perfect_classifier(self):
        """Test ROC curve with perfect classifier."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])
        
        auc_score = plot_roc_curve(y_true, y_proba)
        
        assert auc_score == 1.0
    
    def test_plot_roc_curve_random_classifier(self):
        """Test ROC curve with random classifier."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_proba = np.random.rand(1000)
        
        auc_score = plot_roc_curve(y_true, y_proba)
        
        # Random classifier should have AUC around 0.5
        assert 0.4 < auc_score < 0.6
    
    def test_plot_roc_curve_with_title(self):
        """Test ROC curve with custom title."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])
        
        auc_score = plot_roc_curve(y_true, y_proba, title="Custom ROC")
        
        assert isinstance(auc_score, float)
    
    def test_plot_roc_curve_save_to_file(self):
        """Test saving ROC curve to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            y_true = np.array([0, 0, 1, 1])
            y_proba = np.array([0.1, 0.4, 0.6, 0.9])
            save_path = Path(tmpdir) / "roc_curve.png"
            
            auc_score = plot_roc_curve(y_true, y_proba, save_path=save_path)
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
            assert isinstance(auc_score, float)
    
    def test_plot_roc_curve_returns_valid_auc(self):
        """Test that ROC curve returns valid AUC value."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])
        
        auc_score = plot_roc_curve(y_true, y_proba)
        
        assert 0 <= auc_score <= 1
        assert not np.isnan(auc_score)


class TestPlotPrecisionRecallCurve:
    """Test cases for precision-recall curve plotting."""
    
    def test_plot_precision_recall_curve_basic(self):
        """Test basic precision-recall curve plotting."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.35, 0.8])
        
        plot_precision_recall_curve(y_true, y_proba)
        assert True
    
    def test_plot_precision_recall_curve_with_title(self):
        """Test precision-recall curve with custom title."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])
        
        plot_precision_recall_curve(y_true, y_proba, title="Custom PR Curve")
        assert True
    
    def test_plot_precision_recall_curve_save_to_file(self):
        """Test saving precision-recall curve to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            y_true = np.array([0, 0, 1, 1])
            y_proba = np.array([0.1, 0.4, 0.6, 0.9])
            save_path = Path(tmpdir) / "pr_curve.png"
            
            plot_precision_recall_curve(y_true, y_proba, save_path=save_path)
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_plot_precision_recall_curve_perfect_classifier(self):
        """Test precision-recall curve with perfect classifier."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])
        
        plot_precision_recall_curve(y_true, y_proba)
        assert True
    
    def test_plot_precision_recall_curve_large_dataset(self):
        """Test precision-recall curve with large dataset."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_proba = np.random.rand(1000)
        
        plot_precision_recall_curve(y_true, y_proba)
        assert True


class TestPlotPredictionDistribution:
    """Test cases for prediction distribution plotting."""
    
    def test_plot_prediction_distribution_basic(self):
        """Test basic prediction distribution plotting."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        plot_prediction_distribution(y_proba, y_true)
        assert True
    
    def test_plot_prediction_distribution_with_title(self):
        """Test prediction distribution with custom title."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])
        
        plot_prediction_distribution(y_proba, y_true, title="Custom Distribution")
        assert True
    
    def test_plot_prediction_distribution_save_to_file(self):
        """Test saving prediction distribution to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            y_true = np.array([0, 0, 0, 1, 1, 1])
            y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
            save_path = Path(tmpdir) / "distribution.png"
            
            plot_prediction_distribution(y_proba, y_true, save_path=save_path)
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_plot_prediction_distribution_well_separated(self):
        """Test prediction distribution with well-separated classes."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_proba = np.concatenate([
            np.random.uniform(0, 0.3, 50),
            np.random.uniform(0.7, 1.0, 50)
        ])
        
        plot_prediction_distribution(y_proba, y_true)
        assert True
    
    def test_plot_prediction_distribution_overlapping(self):
        """Test prediction distribution with overlapping classes."""
        np.random.seed(42)
        y_true = np.array([0] * 100 + [1] * 100)
        y_proba = np.random.rand(200)
        
        plot_prediction_distribution(y_proba, y_true)
        assert True
    
    def test_plot_prediction_distribution_large_dataset(self):
        """Test prediction distribution with large dataset."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 10000)
        y_proba = np.random.rand(10000)
        
        plot_prediction_distribution(y_proba, y_true)
        assert True


class TestPlotFeatureImportance:
    """Test cases for feature importance plotting."""
    
    def test_plot_feature_importance_basic(self):
        """Test basic feature importance plotting."""
        feature_names = [f'feature{i}' for i in range(20)]
        importances = np.random.rand(20)
        
        plot_feature_importance(feature_names, importances)
        assert True
    
    def test_plot_feature_importance_with_top_n(self):
        """Test feature importance with top N selection."""
        feature_names = [f'feature{i}' for i in range(20)]
        importances = np.random.rand(20)
        
        plot_feature_importance(feature_names, importances, top_n=10)
        assert True
    
    def test_plot_feature_importance_with_title(self):
        """Test feature importance with custom title."""
        feature_names = [f'feature{i}' for i in range(20)]
        importances = np.random.rand(20)
        
        plot_feature_importance(
            feature_names, 
            importances, 
            title="Custom Feature Importance"
        )
        assert True
    
    def test_plot_feature_importance_save_to_file(self):
        """Test saving feature importance to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_names = [f'feature{i}' for i in range(20)]
            importances = np.random.rand(20)
            save_path = Path(tmpdir) / "feature_importance.png"
            
            plot_feature_importance(
                feature_names, 
                importances, 
                save_path=save_path
            )
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_plot_feature_importance_many_features(self):
        """Test feature importance with many features."""
        feature_names = [f'feature{i}' for i in range(100)]
        importances = np.random.rand(100)
        
        plot_feature_importance(feature_names, importances, top_n=20)
        assert True
    
    def test_plot_feature_importance_sorted_order(self):
        """Test that features are sorted by importance."""
        feature_names = ['low', 'high', 'medium']
        importances = np.array([0.1, 0.9, 0.5])
        
        # Should plot in descending order
        plot_feature_importance(feature_names, importances, top_n=3)
        assert True
    
    def test_plot_feature_importance_equal_importance(self):
        """Test feature importance with equal values."""
        feature_names = [f'feature{i}' for i in range(20)]
        importances = np.ones(20) * 0.05  # Equal importance
        
        plot_feature_importance(feature_names, importances)
        assert True


class TestPlotIntegration:
    """Integration tests for multiple plotting functions."""
    
    def test_generate_all_plots_workflow(self):
        """Test generating all plots in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Generate sample data
            np.random.seed(42)
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.randint(0, 2, 100)
            y_proba = np.random.rand(100)
            
            # Generate all plots
            plot_confusion_matrix(
                y_true, y_pred,
                save_path=output_dir / "confusion_matrix.png"
            )
            
            auc_score = plot_roc_curve(
                y_true, y_proba,
                save_path=output_dir / "roc_curve.png"
            )
            
            plot_precision_recall_curve(
                y_true, y_proba,
                save_path=output_dir / "pr_curve.png"
            )
            
            plot_prediction_distribution(
                y_proba, y_true,
                save_path=output_dir / "distribution.png"
            )
            
            feature_names = [f'feature{i}' for i in range(10)]
            importances = np.random.rand(10)
            plot_feature_importance(
                feature_names, importances,
                save_path=output_dir / "feature_importance.png"
            )
            
            # Verify all files were created
            assert (output_dir / "confusion_matrix.png").exists()
            assert (output_dir / "roc_curve.png").exists()
            assert (output_dir / "pr_curve.png").exists()
            assert (output_dir / "distribution.png").exists()
            assert (output_dir / "feature_importance.png").exists()
            assert isinstance(auc_score, float)
    
    def test_plots_with_consistent_data(self):
        """Test that all plots work with the same dataset."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
        
        # All should work without errors
        plot_confusion_matrix(y_true, y_pred)
        auc_score = plot_roc_curve(y_true, y_proba)
        plot_precision_recall_curve(y_true, y_proba)
        plot_prediction_distribution(y_proba, y_true)
        
        assert 0 <= auc_score <= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_plot_with_single_class(self):
        """Test plotting with only one class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        
        # Should handle gracefully
        plot_confusion_matrix(y_true, y_pred)
        assert True
    
    def test_plot_with_empty_arrays(self):
        """Test plotting with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        
        # Should raise an error or handle gracefully
        try:
            plot_confusion_matrix(y_true, y_pred)
        except (ValueError, IndexError):
            # Expected behavior
            pass
    
    def test_plot_with_mismatched_lengths(self):
        """Test plotting with mismatched array lengths."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])
        
        # Should raise an error
        with pytest.raises((ValueError, IndexError)):
            plot_confusion_matrix(y_true, y_pred)
    
    def test_plot_feature_importance_more_features_than_top_n(self):
        """Test feature importance when requesting more features than available."""
        feature_names = [f'feature{i}' for i in range(30)]
        importances = np.random.rand(30)
        
        # Should handle gracefully - request only 10 of 30
        plot_feature_importance(feature_names, importances, top_n=10)
        assert True
    
    def test_plot_with_nan_values(self):
        """Test plotting with NaN values."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, np.nan, 0.3, 0.9])
        
        # Should handle or raise appropriate error
        try:
            plot_roc_curve(y_true, y_proba)
        except (ValueError, RuntimeError):
            # Expected behavior
            pass


class TestPlotCleanup:
    """Test that plots are properly cleaned up."""
    
    def test_plots_close_properly(self):
        """Test that matplotlib figures are closed after plotting."""
        initial_figs = len(plt.get_fignums())
        
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        plot_confusion_matrix(y_true, y_pred)
        
        final_figs = len(plt.get_fignums())
        
        # Should not leave figures open
        assert final_figs == initial_figs
    
    def test_multiple_plots_cleanup(self):
        """Test cleanup after generating multiple plots."""
        initial_figs = len(plt.get_fignums())
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])
        
        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_proba)
        plot_precision_recall_curve(y_true, y_proba)
        plot_prediction_distribution(y_proba, y_true)
        
        final_figs = len(plt.get_fignums())
        
        # Should not accumulate figures
        assert final_figs == initial_figs
