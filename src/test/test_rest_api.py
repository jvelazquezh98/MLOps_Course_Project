"""
Unit tests for REST API endpoints.

This module contains comprehensive unit tests for all REST API endpoints
defined in the rest_api module, including success cases, error handling,
and edge cases.
"""

import pytest
import json
import io
import tempfile
import pickle
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from fastapi import HTTPException, UploadFile
import pandas as pd

from src.rest_api import (
    root,
    health_check,
    get_project_info,
    get_datasets_info,
    get_available_models,
    validate_with_model
)


class TestRootEndpoint:
    """Test cases for the root endpoint."""
    
    @pytest.mark.asyncio
    async def test_root_returns_correct_response(self):
        """Test that root endpoint returns expected response structure."""
        response = await root()
        
        assert isinstance(response, dict)
        assert "message" in response
        assert "version" in response
        assert "status" in response
        assert response["message"] == "MLOps Course Project API"
        assert response["version"] == "0.1.0"
        assert response["status"] == "running"


class TestHealthCheck:
    """Test cases for the health check endpoint."""
    
    @pytest.mark.asyncio
    async def test_health_check_returns_healthy_status(self):
        """Test that health check returns healthy status."""
        response = await health_check()
        
        assert isinstance(response, dict)
        assert "status" in response
        assert response["status"] == "healthy"


class TestProjectInfo:
    """Test cases for the project info endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_project_info_without_metadata_file(self):
        """Test project info when metadata file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            response = await get_project_info()
        
        assert isinstance(response, dict)
        assert response["project_name"] == "mlops-course-project"
        assert response["python_version"] == "3.13"
        assert "features" in response
        assert "Data Version Control (DVC)" in response["features"]
        assert response["data_engineering_metadata"] == {}
    
    @pytest.mark.asyncio
    async def test_get_project_info_with_valid_metadata_file(self):
        """Test project info when metadata file exists and is valid."""
        mock_metadata = {
            "timestamp": "2025-10-10T15:33:42.103451",
            "dataset_info": {"original_shape": [40436, 62]}
        }
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_metadata))):
            response = await get_project_info()
        
        assert response["data_engineering_metadata"] == mock_metadata
        assert response["project_name"] == "mlops-course-project"
    
    @pytest.mark.asyncio
    async def test_get_project_info_with_invalid_metadata_file(self):
        """Test project info when metadata file exists but is invalid."""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid json")):
            response = await get_project_info()
        
        assert "error" in response["data_engineering_metadata"]
        assert response["data_engineering_metadata"]["error"] == "Could not read metadata file"


class TestDatasetsInfo:
    """Test cases for the datasets info endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_datasets_info_returns_correct_structure(self):
        """Test that datasets info returns expected structure."""
        response = await get_datasets_info()
        
        assert isinstance(response, dict)
        assert "datasets" in response
        assert "data_structure" in response
        
        # Check datasets structure
        assert "online_news_original" in response["datasets"]
        dataset_info = response["datasets"]["online_news_original"]
        assert dataset_info["status"] == "tracked_by_dvc"
        assert dataset_info["format"] == "csv"
        
        # Check data structure
        data_structure = response["data_structure"]
        assert "raw" in data_structure
        assert "interim" in data_structure
        assert "processed" in data_structure


class TestAvailableModels:
    """Test cases for the available models endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_available_models_no_models_directory(self):
        """Test when models directory doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            response = await get_available_models()
        
        assert response["available_models"] == []
        assert response["total_models"] == 0
    
    @pytest.mark.asyncio
    async def test_get_available_models_empty_directory(self):
        """Test when models directory exists but is empty."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.glob.return_value = []
        
        with patch('pathlib.Path', return_value=mock_path):
            response = await get_available_models()
        
        assert response["available_models"] == []
        assert response["total_models"] == 0
    
    @pytest.mark.asyncio
    async def test_get_available_models_with_models(self):
        """Test when models directory contains pickle files."""
        # Mock model files
        mock_model1 = MagicMock()
        mock_model1.stem = "random_forest_model"
        mock_model1.name = "random_forest_model.pkl"
        mock_model1.__str__ = lambda: "models/random_forest_model.pkl"
        
        mock_model2 = MagicMock()
        mock_model2.stem = "xgboost_model"
        mock_model2.name = "xgboost_model.pkl"
        mock_model2.__str__ = lambda: "models/xgboost_model.pkl"
        
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.glob.return_value = [mock_model1, mock_model2]
        
        with patch('pathlib.Path', return_value=mock_path):
            response = await get_available_models()
        
        assert len(response["available_models"]) == 2
        assert response["total_models"] == 2
        
        # Check first model
        model1_info = response["available_models"][0]
        assert model1_info["name"] == "random_forest_model"
        assert model1_info["filename"] == "random_forest_model.pkl"


class TestValidateWithModel:
    """Test cases for the model validation endpoint."""
    
    def create_mock_csv_file(self, content: str, filename: str = "test.csv") -> UploadFile:
        """Helper method to create mock CSV file."""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = filename
        mock_file.read = MagicMock(return_value=content.encode('utf-8'))
        return mock_file
    
    def create_mock_model(self, predictions):
        """Helper method to create mock ML model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = predictions
        return mock_model
    
    @pytest.mark.asyncio
    async def test_validate_with_model_invalid_file_type(self):
        """Test validation with non-CSV file."""
        mock_file = self.create_mock_csv_file("data", "test.txt")
        
        with pytest.raises(HTTPException) as exc_info:
            await validate_with_model(mock_file, "test_model")
        
        assert exc_info.value.status_code == 400
        assert "File must be a CSV file" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_validate_with_model_nonexistent_model(self):
        """Test validation with non-existent model."""
        mock_file = self.create_mock_csv_file("col1,col2\n1,2\n3,4")
        
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await validate_with_model(mock_file, "nonexistent_model")
        
        assert exc_info.value.status_code == 404
        assert "Model 'nonexistent_model' not found" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_validate_with_model_empty_csv(self):
        """Test validation with empty CSV file."""
        mock_file = self.create_mock_csv_file("")
        
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                await validate_with_model(mock_file, "test_model")
        
        assert exc_info.value.status_code == 400
        assert "CSV file is empty" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_validate_with_model_invalid_csv_format(self):
        """Test validation with invalid CSV format."""
        mock_file = self.create_mock_csv_file("invalid,csv,format\n1,2")  # Mismatched columns
        
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                await validate_with_model(mock_file, "test_model")
        
        assert exc_info.value.status_code == 400
        assert "Invalid CSV format" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_validate_with_model_successful_prediction(self):
        """Test successful model validation and prediction."""
        csv_content = "feature1,feature2,feature3\n1,2,3\n4,5,6\n7,8,9"
        mock_file = self.create_mock_csv_file(csv_content)
        
        # Mock predictions
        mock_predictions = [0.8, 0.2, 0.9]
        mock_model = self.create_mock_model(mock_predictions)
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('pickle.load', return_value=mock_model):
            
            response = await validate_with_model(mock_file, "test_model")
        
        assert response["status"] == "success"
        assert response["model_used"] == "test_model"
        assert response["input_shape"] == [3, 3]  # 3 rows, 3 columns
        assert response["predictions_count"] == 3
        assert response["predictions"] == mock_predictions
        assert response["input_columns"] == ["feature1", "feature2", "feature3"]
        assert "Successfully made 3 predictions" in response["message"]
    
    @pytest.mark.asyncio
    async def test_validate_with_model_numpy_predictions(self):
        """Test model validation with numpy array predictions."""
        csv_content = "feature1,feature2\n1,2\n3,4"
        mock_file = self.create_mock_csv_file(csv_content)
        
        # Mock numpy-like predictions with tolist method
        mock_predictions = MagicMock()
        mock_predictions.tolist.return_value = [0.7, 0.3]
        mock_model = self.create_mock_model(mock_predictions)
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('pickle.load', return_value=mock_model):
            
            response = await validate_with_model(mock_file, "test_model")
        
        assert response["predictions"] == [0.7, 0.3]
        assert response["predictions_count"] == 2
    
    @pytest.mark.asyncio
    async def test_validate_with_model_pickle_load_error(self):
        """Test validation when model loading fails."""
        csv_content = "feature1,feature2\n1,2"
        mock_file = self.create_mock_csv_file(csv_content)
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('pickle.load', side_effect=Exception("Model loading failed")):
            
            with pytest.raises(HTTPException) as exc_info:
                await validate_with_model(mock_file, "test_model")
        
        assert exc_info.value.status_code == 500
        assert "Error processing request" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_validate_with_model_prediction_error(self):
        """Test validation when model prediction fails."""
        csv_content = "feature1,feature2\n1,2"
        mock_file = self.create_mock_csv_file(csv_content)
        
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('pickle.load', return_value=mock_model):
            
            with pytest.raises(HTTPException) as exc_info:
                await validate_with_model(mock_file, "test_model")
        
        assert exc_info.value.status_code == 500
        assert "Error processing request" in str(exc_info.value.detail)


# Integration test class for testing multiple endpoints together
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_api_endpoints_return_consistent_types(self):
        """Test that all endpoints return consistent response types."""
        # Test all GET endpoints
        root_response = await root()
        health_response = await health_check()
        project_response = await get_project_info()
        datasets_response = await get_datasets_info()
        models_response = await get_available_models()
        
        # All should return dictionaries
        assert isinstance(root_response, dict)
        assert isinstance(health_response, dict)
        assert isinstance(project_response, dict)
        assert isinstance(datasets_response, dict)
        assert isinstance(models_response, dict)
    
    @pytest.mark.asyncio
    async def test_project_info_contains_expected_features(self):
        """Test that project info contains all expected MLOps features."""
        response = await get_project_info()
        
        expected_features = [
            "Data Version Control (DVC)",
            "ML Pipeline",
            "Model Training",
            "Data Processing"
        ]
        
        for feature in expected_features:
            assert feature in response["features"]