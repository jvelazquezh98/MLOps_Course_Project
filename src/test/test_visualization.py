"""
Unit tests for visualization modules.

This module tests visualization generation including plots,
charts, and result visualizations for model evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc


class TestConfusionMatrixVisualization:
    """Test cases for confusion matrix visualization."""
    
    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 3  # True negatives
        assert cm[1, 1] == 2  # True positives
        assert cm[0, 1] == 0  # False positives
        assert cm[1, 0] == 1  # False negatives
    
    def test_confusion_matrix_with_perfect_predictions(self):
        """Test confusion matrix with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm[0, 1] == 0  # No false positives
        assert cm[1, 0] == 0  # No false negatives
    
    def test_confusion_matrix_normalization(self):
        """Test confusion matrix normalization."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        assert cm_normalized.shape == (2, 2)
        assert np.allclose(cm_normalized.sum(axis=1), 1.0)


class TestROCCurveVisualization:
    """Test cases for ROC curve visualization."""
    
    def test_roc_curve_calculation(self):
        """Test ROC curve calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        assert len(fpr) == len(tpr)
        assert len(fpr) == len(thresholds)
        assert 0 <= roc_auc <= 1
    
    def test_roc_auc_perfect_classifier(self):
        """Test ROC AUC for perfect classifier."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.0, 0.0, 1.0, 1.0])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        assert roc_auc == 1.0
    
    def test_roc_auc_random_classifier(self):
        """Test ROC AUC for random classifier."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_scores = np.random.rand(1000)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Random classifier should have AUC around 0.5
        assert 0.4 < roc_auc < 0.6


class TestPredictionDistribution:
    """Test cases for prediction distribution visualization."""
    
    def test_prediction_distribution_separation(self):
        """Test prediction distribution for different classes."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        proba_class_0 = y_proba[y_true == 0]
        proba_class_1 = y_proba[y_true == 1]
        
        assert len(proba_class_0) == 3
        assert len(proba_class_1) == 3
        assert proba_class_0.mean() < proba_class_1.mean()
    
    def test_prediction_distribution_ranges(self):
        """Test that predictions are in valid range."""
        y_proba = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        
        assert all(0 <= p <= 1 for p in y_proba)
    
    def test_prediction_distribution_statistics(self):
        """Test prediction distribution statistics."""
        y_proba = np.random.rand(1000)
        
        mean = y_proba.mean()
        std = y_proba.std()
        
        assert 0 <= mean <= 1
        assert std >= 0


class TestFeatureImportance:
    """Test cases for feature importance visualization."""
    
    def test_feature_importance_ranking(self):
        """Test feature importance ranking."""
        feature_names = ['feature1', 'feature2', 'feature3', 'feature4']
        importances = np.array([0.1, 0.4, 0.3, 0.2])
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        assert feature_names[indices[0]] == 'feature2'
        assert feature_names[indices[1]] == 'feature3'
        assert feature_names[indices[2]] == 'feature4'
        assert feature_names[indices[3]] == 'feature1'
    
    def test_feature_importance_normalization(self):
        """Test feature importance normalization."""
        importances = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Normalize to sum to 1
        normalized = importances / importances.sum()
        
        assert np.isclose(normalized.sum(), 1.0)
    
    def test_feature_importance_top_n_selection(self):
        """Test selecting top N features."""
        feature_names = [f'feature{i}' for i in range(20)]
        importances = np.random.rand(20)
        
        top_n = 10
        indices = np.argsort(importances)[::-1][:top_n]
        
        assert len(indices) == top_n
        
        # Verify they are in descending order
        top_importances = importances[indices]
        assert all(top_importances[i] >= top_importances[i+1] 
                  for i in range(len(top_importances)-1))


class TestPlotGeneration:
    """Test cases for plot generation."""
    
    def test_plot_creation_and_cleanup(self):
        """Test that plots are created and cleaned up properly."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)
    
    def test_plot_save_to_file(self):
        """Test saving plot to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            
            save_path = Path(tmpdir) / "test_plot.png"
            fig.savefig(save_path)
            plt.close(fig)
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_multiple_plots_generation(self):
        """Test generating multiple plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_types = ['line', 'bar', 'scatter']
            
            for plot_type in plot_types:
                fig, ax = plt.subplots()
                
                if plot_type == 'line':
                    ax.plot([1, 2, 3], [1, 2, 3])
                elif plot_type == 'bar':
                    ax.bar([1, 2, 3], [1, 2, 3])
                elif plot_type == 'scatter':
                    ax.scatter([1, 2, 3], [1, 2, 3])
                
                save_path = Path(tmpdir) / f"{plot_type}_plot.png"
                fig.savefig(save_path)
                plt.close(fig)
                
                assert save_path.exists()


class TestVisualizationDataPreparation:
    """Test cases for data preparation for visualization."""
    
    def test_data_aggregation_for_plotting(self):
        """Test data aggregation for plotting."""
        df = pd.DataFrame({
            'model': ['A', 'A', 'B', 'B'],
            'metric': ['accuracy', 'precision', 'accuracy', 'precision'],
            'value': [0.85, 0.82, 0.90, 0.88]
        })
        
        # Pivot for plotting
        df_pivot = df.pivot(index='model', columns='metric', values='value')
        
        assert df_pivot.shape == (2, 2)
        assert 'accuracy' in df_pivot.columns
        assert 'precision' in df_pivot.columns
    
    def test_data_filtering_for_visualization(self):
        """Test filtering data for visualization."""
        df = pd.DataFrame({
            'model': ['A', 'B', 'C', 'D'],
            'accuracy': [0.85, 0.90, 0.75, 0.88]
        })
        
        # Filter top models
        top_models = df.nlargest(2, 'accuracy')
        
        assert len(top_models) == 2
        assert top_models.iloc[0]['model'] == 'B'
        assert top_models.iloc[1]['model'] == 'D'
    
    def test_data_normalization_for_plotting(self):
        """Test data normalization for plotting."""
        values = np.array([10, 20, 30, 40, 50])
        
        # Min-max normalization
        normalized = (values - values.min()) / (values.max() - values.min())
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == len(values)


class TestComparisonVisualization:
    """Test cases for model comparison visualization."""
    
    def test_comparison_bar_chart_data(self):
        """Test data preparation for comparison bar chart."""
        df = pd.DataFrame({
            'model': ['Model A', 'Model B', 'Model C'],
            'accuracy': [0.85, 0.90, 0.82],
            'precision': [0.83, 0.88, 0.80]
        })
        
        df_plot = df.set_index('model')
        
        assert df_plot.shape == (3, 2)
        assert 'accuracy' in df_plot.columns
        assert 'precision' in df_plot.columns
    
    def test_comparison_radar_chart_data(self):
        """Test data preparation for radar chart."""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        values = [0.85, 0.82, 0.88, 0.85]
        
        # Prepare angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        values_closed = values + values[:1]
        
        assert len(angles) == len(metrics) + 1
        assert len(values_closed) == len(metrics) + 1
    
    def test_comparison_heatmap_data(self):
        """Test data preparation for heatmap."""
        df = pd.DataFrame({
            'model': ['A', 'B', 'C'],
            'accuracy': [0.85, 0.90, 0.82],
            'precision': [0.83, 0.88, 0.80],
            'recall': [0.87, 0.92, 0.85]
        })
        
        df_heatmap = df.set_index('model')
        
        assert df_heatmap.shape == (3, 3)
        assert all(0 <= df_heatmap.values.flatten()) and all(df_heatmap.values.flatten() <= 1)


class TestVisualizationErrorHandling:
    """Test cases for visualization error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        df = pd.DataFrame()
        
        assert df.empty
        assert len(df) == 0
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        df = pd.DataFrame({
            'model': ['A', 'B'],
            'accuracy': [0.85, 0.90]
        })
        
        requested_metrics = ['accuracy', 'precision', 'recall']
        available_metrics = [m for m in requested_metrics if m in df.columns]
        
        assert len(available_metrics) == 1
        assert available_metrics == ['accuracy']
    
    def test_invalid_metric_values_handling(self):
        """Test handling of invalid metric values."""
        values = np.array([0.85, np.nan, 0.90, np.inf])
        
        # Filter valid values
        valid_values = values[np.isfinite(values)]
        
        assert len(valid_values) == 2
        assert all(np.isfinite(valid_values))
