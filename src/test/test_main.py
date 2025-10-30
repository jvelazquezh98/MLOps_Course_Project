"""
Integration tests for the FastAPI application.

This module contains integration tests that test the complete FastAPI application
including routing, middleware, and end-to-end functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, mock_open, MagicMock
import json
import io

from src.main import app


class TestFastAPIApplication:
    """Integration tests for the FastAPI application."""
    
    def setup_method(self):
        """Set up test client for each test."""
        self.client = TestClient(app)
    
    def test_root_endpoint_integration(self):
        """Test root endpoint through FastAPI application."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "MLOps Course Project API"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"
    
    def test_health_endpoint_integration(self):
        """Test health endpoint through FastAPI application."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_project_info_endpoint_integration(self):
        """Test project info endpoint through FastAPI application."""
        with patch('os.path.exists', return_value=False):
            response = self.client.get("/project-info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["project_name"] == "mlops-course-project"
        assert "features" in data
        assert isinstance(data["features"], list)
    
    def test_datasets_endpoint_integration(self):
        """Test datasets endpoint through FastAPI application."""
        response = self.client.get("/datasets")
        
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "data_structure" in data
        assert "online_news_original" in data["datasets"]
    
    def test_models_endpoint_integration(self):
        """Test models endpoint through FastAPI application."""
        with patch('pathlib.Path.exists', return_value=False):
            response = self.client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "total_models" in data
        assert data["total_models"] == 0
    
    def test_validate_endpoint_missing_file(self):
        """Test validate endpoint with missing file."""
        response = self.client.post("/validate", data={"model_name": "test_model"})
        
        assert response.status_code == 422  # Validation error for missing file
    
    def test_validate_endpoint_missing_model_name(self):
        """Test validate endpoint with missing model name."""
        files = {"csv_file": ("test.csv", "col1,col2\n1,2", "text/csv")}
        response = self.client.post("/validate", files=files)
        
        assert response.status_code == 422  # Validation error for missing model_name
    
    def test_validate_endpoint_invalid_file_type(self):
        """Test validate endpoint with invalid file type."""
        files = {"csv_file": ("test.txt", "some content", "text/plain")}
        data = {"model_name": "test_model"}
        
        response = self.client.post("/validate", files=files, data=data)
        
        assert response.status_code == 400
        assert "File must be a CSV file" in response.json()["detail"]
    
    def test_validate_endpoint_nonexistent_model(self):
        """Test validate endpoint with non-existent model."""
        files = {"csv_file": ("test.csv", "col1,col2\n1,2", "text/csv")}
        data = {"model_name": "nonexistent_model"}
        
        with patch('pathlib.Path.exists', return_value=False):
            response = self.client.post("/validate", files=files, data=data)
        
        assert response.status_code == 404
        assert "Model 'nonexistent_model' not found" in response.json()["detail"]
    
    def test_validate_endpoint_successful_prediction(self):
        """Test successful prediction through validate endpoint."""
        csv_content = "feature1,feature2\n1,2\n3,4"
        files = {"csv_file": ("test.csv", csv_content, "text/csv")}
        data = {"model_name": "test_model"}
        
        # Mock successful model loading and prediction
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.2]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('pickle.load', return_value=mock_model):
            
            response = self.client.post("/validate", files=files, data=data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_used"] == "test_model"
        assert data["predictions"] == [0.8, 0.2]
        assert data["predictions_count"] == 2
    
    def test_openapi_schema_generation(self):
        """Test that OpenAPI schema is generated correctly."""
        response = self.client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "MLOps Course Project API"
        assert schema["info"]["version"] == "0.1.0"
    
    def test_docs_endpoint_accessible(self):
        """Test that API documentation is accessible."""
        response = self.client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint_accessible(self):
        """Test that ReDoc documentation is accessible."""
        response = self.client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_nonexistent_endpoint_returns_404(self):
        """Test that non-existent endpoints return 404."""
        response = self.client.get("/nonexistent")
        
        assert response.status_code == 404
    
    def test_api_endpoints_have_correct_tags(self):
        """Test that API endpoints are properly tagged in OpenAPI schema."""
        response = self.client.get("/openapi.json")
        schema = response.json()
        
        # Check that endpoints have appropriate tags
        paths = schema["paths"]
        
        # Root endpoint should have "General" tag
        assert "General" in [tag["name"] for tag in paths["/"]["get"]["tags"]]
        
        # Health endpoint should have "General" tag
        assert "General" in [tag["name"] for tag in paths["/health"]["get"]["tags"]]
        
        # Project info should have "Project" tag
        assert "Project" in [tag["name"] for tag in paths["/project-info"]["get"]["tags"]]
        
        # Datasets should have "Data" tag
        assert "Data" in [tag["name"] for tag in paths["/datasets"]["get"]["tags"]]
        
        # Models should have "Models" tag
        assert "Models" in [tag["name"] for tag in paths["/models"]["get"]["tags"]]
        
        # Validate should have "Prediction" tag
        assert "Prediction" in [tag["name"] for tag in paths["/validate"]["post"]["tags"]]


class TestAPIErrorHandling:
    """Test error handling across the API."""
    
    def setup_method(self):
        """Set up test client for each test."""
        self.client = TestClient(app)
    
    def test_internal_server_error_handling(self):
        """Test that internal server errors are handled gracefully."""
        files = {"csv_file": ("test.csv", "col1,col2\n1,2", "text/csv")}
        data = {"model_name": "test_model"}
        
        # Mock an internal server error during model loading
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', side_effect=Exception("Internal error")):
            
            response = self.client.post("/validate", files=files, data=data)
        
        assert response.status_code == 500
        assert "Error processing request" in response.json()["detail"]
    
    def test_malformed_request_handling(self):
        """Test handling of malformed requests."""
        # Send invalid JSON in request body
        response = self.client.post(
            "/validate",
            headers={"Content-Type": "application/json"},
            data="invalid json"
        )
        
        # Should return validation error
        assert response.status_code == 422


class TestAPIPerformance:
    """Basic performance tests for API endpoints."""
    
    def setup_method(self):
        """Set up test client for each test."""
        self.client = TestClient(app)
    
    def test_multiple_concurrent_requests(self):
        """Test that API can handle multiple concurrent requests."""
        import concurrent.futures
        import time
        
        def make_request():
            start_time = time.time()
            response = self.client.get("/health")
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for status_code, duration in results:
            assert status_code == 200
            assert duration < 1.0  # Should respond within 1 second