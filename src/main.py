"""
MLOps Course Project - FastAPI Application Entry Point

This module serves as the main entry point for the MLOps Course Project REST API.
It initializes the FastAPI application and registers all API endpoints defined
in the rest_api module.

The API provides endpoints for:
- Project information and metadata
- Dataset information
- ML model management
- Model prediction/validation services

Usage:
    Run the application using uvicorn:
    $ uvicorn src.main:app --reload

Author: MLOps Course Project Team
Version: 0.1.0
"""

from fastapi import FastAPI
from src.rest_api import (
    root,
    health_check,
    get_project_info,
    get_datasets_info,
    get_available_models,
    validate_with_model
)

# Initialize FastAPI application with metadata
app = FastAPI(
    title="MLOps Course Project API",
    description="REST API for the MLOps course project demonstrating ML pipeline capabilities",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Register API endpoints with their respective HTTP methods and routes
app.get("/", summary="Root endpoint", tags=["General"])(root)
app.get("/health", summary="Health check", tags=["General"])(health_check)
app.get("/project-info", summary="Get project information", tags=["Project"])(get_project_info)
app.get("/datasets", summary="Get datasets information", tags=["Data"])(get_datasets_info)
app.get("/models", summary="Get available models", tags=["Models"])(get_available_models)
app.post("/validate", summary="Validate data with ML model", tags=["Prediction"])(validate_with_model)