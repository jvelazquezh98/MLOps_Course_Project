"""
MLOps Course Project - REST API Endpoints

This module contains all the REST API endpoint implementations for the MLOps Course Project.
It provides functionality for project information, dataset management, model operations,
and ML model prediction services.

The module includes endpoints for:
- Basic API information and health checks
- Project metadata and configuration details
- Dataset information and structure
- Available ML models listing
- Model prediction and validation services

Dependencies:
    - FastAPI for web framework functionality
    - pandas for data manipulation
    - pickle for model serialization
    - pathlib for file system operations

Author: MLOps Course Project Team
Version: 0.1.0
"""

from fastapi import UploadFile, File, Form, HTTPException
from typing import Dict, Any, Optional
import json
import os
import pandas as pd
import pickle
import io
from pathlib import Path
from loguru import logger
import sys

# Import training function from modeling module
sys.path.append(str(Path(__file__).parent))
from modeling.train import train_model, DEFAULT_PARAMS, ALLOWED_PARAM_KEYS


async def root() -> Dict[str, str]:
    """
    Root endpoint providing basic API information.
    
    This endpoint serves as the main entry point for the API, returning
    basic information about the service including name, version, and status.
    
    Returns:
        Dict[str, str]: Dictionary containing API metadata
            - message: API name and description
            - version: Current API version
            - status: Current operational status
    
    Example:
        GET /
        Response: {
            "message": "MLOps Course Project API",
            "version": "0.1.0",
            "status": "running"
        }
    """
    return {
        "message": "MLOps Course Project API",
        "version": "0.1.0",
        "status": "running"
    }


async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for monitoring service availability.
    
    This endpoint is used by monitoring systems and load balancers to verify
    that the API service is running and responsive.
    
    Returns:
        Dict[str, str]: Health status information
            - status: Current health status ("healthy" if operational)
    
    Example:
        GET /health
        Response: {"status": "healthy"}
    """
    return {"status": "healthy"}


async def get_project_info() -> Dict[str, Any]:
    """
    Retrieve comprehensive project information and metadata.
    
    This endpoint provides detailed information about the MLOps project including
    configuration details, data engineering metadata, and available features.
    It attempts to read additional metadata from the data engineering JSON file
    if available.
    
    Returns:
        Dict[str, Any]: Comprehensive project information
            - project_name: Name of the MLOps project
            - description: Project description and purpose
            - python_version: Python version requirement
            - data_engineering_metadata: Metadata from data engineering processes
            - features: List of implemented MLOps features
    
    Raises:
        No exceptions are raised; errors in reading metadata files are handled gracefully
    
    Example:
        GET /project-info
        Response: {
            "project_name": "mlops-course-project",
            "description": "MLOps Course Project demonstrating ML pipeline capabilities",
            "python_version": "3.13",
            "data_engineering_metadata": {...},
            "features": ["Data Version Control (DVC)", "ML Pipeline", ...]
        }
    """
    
    # Try to read the data engineering metadata if it exists
    metadata_file = "01_data_engineering_metadata_JLRL.json"
    metadata = {}
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = {"error": "Could not read metadata file"}
    
    return {
        "project_name": "mlops-course-project",
        "description": "MLOps Course Project demonstrating ML pipeline capabilities",
        "python_version": "3.13",
        "data_engineering_metadata": metadata,
        "features": [
            "Data Version Control (DVC)",
            "ML Pipeline",
            "Model Training",
            "Data Processing"
        ]
    }


async def get_datasets_info() -> Dict[str, Any]:
    """
    Retrieve information about available datasets and data structure.
    
    This endpoint provides comprehensive information about the datasets used in
    the MLOps project, including their current status, format, and the overall
    data pipeline structure.
    
    Returns:
        Dict[str, Any]: Dataset information and structure
            - datasets: Dictionary of available datasets with metadata
                - status: Current tracking status (e.g., "tracked_by_dvc")
                - format: Data format (e.g., "csv")
                - description: Human-readable description
            - data_structure: Explanation of data pipeline stages
                - raw: Original unmodified data
                - interim: Intermediate processing stage
                - processed: Final modeling-ready data
    
    Example:
        GET /datasets
        Response: {
            "datasets": {
                "online_news_original": {
                    "status": "tracked_by_dvc",
                    "format": "csv",
                    "description": "Original online news dataset"
                }
            },
            "data_structure": {
                "raw": "Original unmodified data",
                "interim": "Intermediate data during cleaning/transformation",
                "processed": "Final data ready for modeling"
            }
        }
    """
    return {
        "datasets": {
            "online_news_original": {
                "status": "tracked_by_dvc",
                "format": "csv",
                "description": "Original online news dataset"
            }
        },
        "data_structure": {
            "raw": "Original unmodified data",
            "interim": "Intermediate data during cleaning/transformation", 
            "processed": "Final data ready for modeling"
        }
    }


async def get_available_models() -> Dict[str, Any]:
    """
    Retrieve list of available trained ML models.
    
    This endpoint scans the models directory for pickle files (.pkl) and returns
    information about all available trained models that can be used for predictions.
    
    Returns:
        Dict[str, Any]: Information about available models
            - available_models: List of model dictionaries, each containing:
                - name: Model name (filename without extension)
                - filename: Full filename including extension
                - path: Relative path to the model file
            - total_models: Total count of available models
    
    Example:
        GET /models
        Response: {
            "available_models": [
                {
                    "name": "random_forest_model",
                    "filename": "random_forest_model.pkl",
                    "path": "models/random_forest_model.pkl"
                }
            ],
            "total_models": 1
        }
    """
    models_dir = Path("models")
    available_models = []
    
    if models_dir.exists():
        # change for to loop over folders inside models_dir
        for model_folder in models_dir.iterdir():
            if model_folder.is_dir():
                for model_file in model_folder.glob("*.pkl"):
                    available_models.append({
                        "name": model_folder.name,
                        "filename": model_file.name,
                        "path": str(model_file)
                    })
    
    return {
        "available_models": available_models,
        "total_models": len(available_models)
    }


async def validate_with_model(
    csv_file: UploadFile = File(..., description="CSV file to make predictions on"),
    model_name: str = Form(..., description="Name of the model to use for prediction")
) -> Dict[str, Any]:
    """
    Validate and make predictions using a trained ML model on uploaded CSV data.
    
    This endpoint accepts a CSV file and a model name, loads the specified trained
    model from the models directory, and returns predictions along with metadata
    about the prediction process.
    
    Args:
        csv_file (UploadFile): CSV file containing the data for prediction.
            Must be a valid CSV file with appropriate columns for the model.
        model_name (str): Name of the model to use (without .pkl extension).
            Must correspond to an existing pickle file in the models directory.
    
    Returns:
        Dict[str, Any]: Prediction results and metadata
            - status: Success/failure status of the operation
            - model_used: Name of the model that was used
            - input_shape: Shape of the input data (rows, columns)
            - predictions_count: Number of predictions made
            - predictions: List of prediction values
            - input_columns: List of column names from input data
            - message: Human-readable success message
    
    Raises:
        HTTPException: 
            - 400: If file is not CSV format, empty, or has parsing errors
            - 404: If specified model is not found
            - 500: If there's an error during model loading or prediction
    
    Example:
        POST /validate
        Form data: 
            - csv_file: data.csv
            - model_name: "random_forest_model"
        
        Response: {
            "status": "success",
            "model_used": "random_forest_model",
            "input_shape": [100, 10],
            "predictions_count": 100,
            "predictions": [0.8, 0.2, 0.9, ...],
            "input_columns": ["feature1", "feature2", ...],
            "message": "Successfully made 100 predictions using model 'random_forest_model'"
        }
    """
    
    # Validate file type
    if not csv_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    # Check if model exists
    model_path = Path(f"models/{model_name}/model.pkl")
    if not model_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available models can be checked at /models endpoint"
        )
    
    try:
        # Read CSV file
        contents = await csv_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Store original columns for reference
        original_columns = df.columns.tolist()
        
        # Remove non-numeric columns (like 'url', text fields, etc.)
        # Also remove target variable if present
        columns_to_drop = ['url', 'shares']
        df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        # Ensure all remaining columns are numeric
        df_clean = df_clean.select_dtypes(include=['number'])
        print(df_clean)
        
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Get expected features from the model if available
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_.tolist()
            logger.info(f"Model expects {len(expected_features)} features: {expected_features}")
            logger.info(f"CSV has {len(df_clean.columns)} features: {df_clean.columns.tolist()}")
            
            # Filter dataframe to only include expected features
            missing_features = [f for f in expected_features if f not in df_clean.columns]
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required features: {missing_features}. Model expects: {expected_features}"
                )
            df_clean = df_clean[expected_features]
        else:
            logger.warning(f"Model doesn't have feature_names_in_ attribute. Using all {len(df_clean.columns)} numeric columns.")
        
        # Make predictions
        predictions = model.predict(df_clean.values)
        
        # Convert predictions to list for JSON serialization
        if hasattr(predictions, 'tolist'):
            predictions_list = predictions.tolist()
        else:
            predictions_list = list(predictions)
        
        return {
            "status": "success",
            "model_used": model_name,
            "input_shape": df.shape,
            "features_used": df_clean.shape[1],
            "predictions_count": len(predictions_list),
            "predictions": predictions_list,
            "original_columns": original_columns,
            "features_columns": df_clean.columns.tolist(),
            "message": f"Successfully made {len(predictions_list)} predictions using model '{model_name}'"
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )


async def train_test(
    data_path: Optional[str] = Form(None, description="Path to the training dataset CSV file"),
    target: Optional[str] = Form(None, description="Name of the target column"),
    params: Optional[str] = Form(None, description="JSON string with model hyperparameters"),
) -> Dict[str, Any]:
    """
    Train a Random Forest model with the provided dataset and parameters.
    
    This endpoint accepts training parameters and executes the model training process
    using MLflow for experiment tracking. It trains a RandomForestClassifier and
    registers the model in MLflow.
    
    Args:
        data_path (str): Path to the CSV file containing training data.
            Must be a valid path to an existing CSV file.
        target (str): Name of the target column in the dataset.
            If the column doesn't exist but 'shares' exists, it will be created
            automatically as a binary classification (shares > 1400).
        params (Optional[str]): JSON string with model hyperparameters.
            Allowed parameters: n_estimators, max_depth, min_samples_split,
            min_samples_leaf, max_features, random_state.
            Example: '{"n_estimators": 300, "max_depth": 12}'
    """

    function_result = train_model(data_path=Path(data_path) if data_path else None
                , target=target if target else None
                , params=json.loads(params) if params else None
                , save_metrics=False)
    
    return {
        "message": "Model training test completed successfully.",
        "metrics": function_result["metrics"]
    }
