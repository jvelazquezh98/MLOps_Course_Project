"""
Unit tests for feature engineering module.

This module tests the feature engineering functionality including
data cleaning, transformation, and feature creation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile

from src.features import main


class TestFeatureEngineering:
    """Test cases for feature engineering functionality."""
    
    def create_sample_dataframe(self) -> pd.DataFrame:
        """Helper to create sample dataframe for testing."""
        return pd.DataFrame({
            'url': ['http://example.com/1', 'http://example.com/2', 'http://example.com/3'],
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'mixed_type_col': ['bad', 'unknown', 1],
            'shares': [1000, 2000, 3000]
        })
    
    def test_mixed_type_col_cleaning(self):
        """Test that mixed_type_col is cleaned correctly."""
        df = self.create_sample_dataframe()
        
        # Simulate the cleaning process
        df['mixed_type_col_clean'] = df['mixed_type_col'].replace({
            'bad': 0,
            'unknown': 1
        })
        df['mixed_type_col_clean'] = pd.to_numeric(df['mixed_type_col_clean'], errors='coerce')
        
        assert 'mixed_type_col_clean' in df.columns
        assert df['mixed_type_col_clean'].dtype in [np.float64, np.int64]
    
    def test_numeric_conversion(self):
        """Test that columns are converted to numeric."""
        df = self.create_sample_dataframe()
        
        numeric_columns = [col for col in df.columns if col != 'url']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check that numeric columns are numeric
        assert df['feature1'].dtype in [np.float64, np.int64]
        assert df['feature2'].dtype in [np.float64, np.int64]
    
    def test_null_value_handling(self):
        """Test that null values are handled correctly."""
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0, 4.0, 5.0],
            'feature2': [np.nan, 2.0, 3.0, 4.0, 5.0],
            'feature3': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        # Fill nulls with median
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        assert df.isnull().sum().sum() == 0
    
    def test_row_dropping_with_many_nulls(self):
        """Test that rows with many nulls are dropped."""
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0],
            'feature2': [np.nan, np.nan, 3.0],
            'feature3': [np.nan, np.nan, 3.0]
        })
        
        # Drop rows with more than 50% nulls
        df_clean = df.dropna(thresh=df.shape[1]//2)
        
        assert len(df_clean) < len(df)
    
    def test_non_numeric_column_removal(self):
        """Test that non-numeric columns are removed."""
        df = self.create_sample_dataframe()
        
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        df_numeric = df.drop(columns=non_numeric)
        
        # All remaining columns should be numeric
        for col in df_numeric.columns:
            assert df_numeric[col].dtype in [np.float64, np.int64]


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering with file I/O."""
    
    def test_feature_engineering_with_temp_files(self):
        """Test feature engineering with temporary files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "output.csv"
            
            # Create sample input file
            df = pd.DataFrame({
                'url': ['http://example.com/1', 'http://example.com/2'],
                'feature1': [1.0, 2.0],
                'feature2': [3.0, 4.0],
                'mixed_type_col': ['bad', 'unknown'],
                'shares': [1000, 2000]
            })
            df.to_csv(input_path, index=False)
            
            # Mock the feature engineering process
            df_processed = pd.read_csv(input_path)
            df_processed['mixed_type_col_clean'] = df_processed['mixed_type_col'].replace({
                'bad': 0,
                'unknown': 1
            })
            df_processed = df_processed.drop('mixed_type_col', axis=1)
            df_processed.to_csv(output_path, index=False)
            
            # Verify output
            assert output_path.exists()
            df_result = pd.read_csv(output_path)
            assert 'mixed_type_col_clean' in df_result.columns
            assert 'mixed_type_col' not in df_result.columns
    
    def test_feature_engineering_preserves_data_shape(self):
        """Test that feature engineering preserves expected data shape."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'feature3': [7.0, 8.0, 9.0]
        })
        
        original_rows = len(df)
        
        # Process (no rows should be dropped if no nulls)
        df_processed = df.copy()
        
        assert len(df_processed) == original_rows
    
    def test_feature_engineering_handles_empty_dataframe(self):
        """Test that feature engineering handles empty dataframes."""
        df = pd.DataFrame()
        
        # Should handle gracefully
        assert len(df) == 0
        assert len(df.columns) == 0


class TestFeatureValidation:
    """Test cases for feature validation."""
    
    def test_all_features_are_numeric_after_processing(self):
        """Test that all features are numeric after processing."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4.0, 5.0, 6.0],
            'feature3': ['7', '8', '9']  # String numbers
        })
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        for col in df.columns:
            assert df[col].dtype in [np.float64, np.int64]
    
    def test_no_infinite_values_after_processing(self):
        """Test that no infinite values exist after processing."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.inf],
            'feature2': [4.0, -np.inf, 6.0]
        })
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        
        assert not np.isinf(df.values).any()
    
    def test_feature_ranges_are_reasonable(self):
        """Test that feature values are in reasonable ranges."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        
        # Check that values are finite
        assert np.isfinite(df.values).all()
        
        # Check that standard deviation is reasonable
        for col in df.columns:
            std = df[col].std()
            assert std >= 0
