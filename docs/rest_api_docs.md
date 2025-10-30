# MLOps Course Project - REST API Documentation

## Overview

The MLOps Course Project REST API provides a comprehensive interface for interacting with machine learning models, datasets, and project metadata. This API is built using FastAPI and offers endpoints for project information, dataset management, model operations, and ML model prediction services.

**Base URL:** `http://localhost:8000` (when running locally)

**API Version:** 0.1.0

**Documentation URLs:**
- Interactive API Documentation: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

## Architecture

The API is structured with the following components:

- **FastAPI Application** (`src/main.py`): Main application entry point with endpoint registration
- **REST API Endpoints** (`src/rest_api.py`): Implementation of all API endpoint logic
- **Dependencies**: FastAPI, pandas, scikit-learn, pickle, pathlib

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

## Content Types

- **Request Content Types:** `application/json`, `multipart/form-data`
- **Response Content Type:** `application/json`

## API Endpoints

### 1. Root Endpoint

**Endpoint:** `GET /`  
**Tags:** General  
**Summary:** Root endpoint providing basic API information

#### Description
This endpoint serves as the main entry point for the API, returning basic information about the service including name, version, and status.

#### Response
```json
{
  "message": "MLOps Course Project API",
  "version": "0.1.0",
  "status": "running"
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `message` | string | API name and description |
| `version` | string | Current API version |
| `status` | string | Current operational status |

---

### 2. Health Check

**Endpoint:** `GET /health`  
**Tags:** General  
**Summary:** Health check endpoint for monitoring service availability

#### Description
This endpoint is used by monitoring systems and load balancers to verify that the API service is running and responsive.

#### Response
```json
{
  "status": "healthy"
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Current health status ("healthy" if operational) |

---

### 3. Project Information

**Endpoint:** `GET /project-info`  
**Tags:** Project  
**Summary:** Retrieve comprehensive project information and metadata

#### Description
This endpoint provides detailed information about the MLOps project including configuration details, data engineering metadata, and available features. It attempts to read additional metadata from the data engineering JSON file if available.

#### Response
```json
{
  "project_name": "mlops-course-project",
  "description": "MLOps Course Project demonstrating ML pipeline capabilities",
  "python_version": "3.13",
  "data_engineering_metadata": {
    "timestamp": "2025-10-10T15:33:42.103451",
    "data_engineer_activities": {
      "outlier_treatment": true,
      "missing_value_imputation": true,
      "categorical_encoding": true,
      "feature_engineering": true,
      "pipeline_creation": true,
      "dimensionality_reduction": true
    },
    "dataset_info": {
      "original_shape": [40436, 62],
      "engineered_shape": [40436, 68],
      "processed_shape": [40436, 34]
    }
  },
  "features": [
    "Data Version Control (DVC)",
    "ML Pipeline",
    "Model Training",
    "Data Processing"
  ]
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `project_name` | string | Name of the MLOps project |
| `description` | string | Project description and purpose |
| `python_version` | string | Python version requirement |
| `data_engineering_metadata` | object | Metadata from data engineering processes |
| `features` | array[string] | List of implemented MLOps features |

#### Error Handling
- Errors in reading metadata files are handled gracefully
- If metadata file cannot be read, returns `{"error": "Could not read metadata file"}`

---

### 4. Dataset Information

**Endpoint:** `GET /datasets`  
**Tags:** Data  
**Summary:** Retrieve information about available datasets and data structure

#### Description
This endpoint provides comprehensive information about the datasets used in the MLOps project, including their current status, format, and the overall data pipeline structure.

#### Response
```json
{
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
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `datasets` | object | Dictionary of available datasets with metadata |
| `datasets.*.status` | string | Current tracking status (e.g., "tracked_by_dvc") |
| `datasets.*.format` | string | Data format (e.g., "csv") |
| `datasets.*.description` | string | Human-readable description |
| `data_structure` | object | Explanation of data pipeline stages |
| `data_structure.raw` | string | Original unmodified data description |
| `data_structure.interim` | string | Intermediate processing stage description |
| `data_structure.processed` | string | Final modeling-ready data description |

---

### 5. Available Models

**Endpoint:** `GET /models`  
**Tags:** Models  
**Summary:** Retrieve list of available trained ML models

#### Description
This endpoint scans the models directory for pickle files (.pkl) and returns information about all available trained models that can be used for predictions.

#### Response
```json
{
  "available_models": [
    {
      "name": "random_forest_model",
      "filename": "random_forest_model.pkl",
      "path": "models/random_forest_model.pkl"
    },
    {
      "name": "xgboost_model",
      "filename": "xgboost_model.pkl",
      "path": "models/xgboost_model.pkl"
    }
  ],
  "total_models": 2
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `available_models` | array[object] | List of model dictionaries |
| `available_models[].name` | string | Model name (filename without extension) |
| `available_models[].filename` | string | Full filename including extension |
| `available_models[].path` | string | Relative path to the model file |
| `total_models` | integer | Total count of available models |

---

### 6. Model Validation/Prediction

**Endpoint:** `POST /validate`  
**Tags:** Prediction  
**Summary:** Validate and make predictions using a trained ML model on uploaded CSV data

#### Description
This endpoint accepts a CSV file and a model name, loads the specified trained model from the models directory, and returns predictions along with metadata about the prediction process.

#### Request Parameters
**Content Type:** `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `csv_file` | file | Yes | CSV file containing the data for prediction. Must be a valid CSV file with appropriate columns for the model. |
| `model_name` | string | Yes | Name of the model to use (without .pkl extension). Must correspond to an existing pickle file in the models directory. |

#### Request Example
```bash
curl -X POST "http://localhost:8000/validate" \
  -F "csv_file=@data.csv" \
  -F "model_name=random_forest_model"
```

#### Success Response (200)
```json
{
  "status": "success",
  "model_used": "random_forest_model",
  "input_shape": [100, 10],
  "predictions_count": 100,
  "predictions": [0.8, 0.2, 0.9, 0.7, 0.1],
  "input_columns": ["feature1", "feature2", "feature3", "feature4", "feature5"],
  "message": "Successfully made 100 predictions using model 'random_forest_model'"
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Success/failure status of the operation |
| `model_used` | string | Name of the model that was used |
| `input_shape` | array[integer] | Shape of the input data [rows, columns] |
| `predictions_count` | integer | Number of predictions made |
| `predictions` | array[number] | List of prediction values |
| `input_columns` | array[string] | List of column names from input data |
| `message` | string | Human-readable success message |

#### Error Responses

**400 Bad Request - Invalid File Type**
```json
{
  "detail": "File must be a CSV file"
}
```

**400 Bad Request - Empty CSV File**
```json
{
  "detail": "CSV file is empty"
}
```

**400 Bad Request - Invalid CSV Format**
```json
{
  "detail": "Invalid CSV format"
}
```

**404 Not Found - Model Not Found**
```json
{
  "detail": "Model 'nonexistent_model' not found. Available models can be checked at /models endpoint"
}
```

**422 Unprocessable Entity - Missing Parameters**
```json
{
  "detail": [
    {
      "loc": ["body", "csv_file"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**500 Internal Server Error - Processing Error**
```json
{
  "detail": "Error processing request: [specific error message]"
}
```

## Usage Examples

### Python with requests
```python
import requests

# Get project information
response = requests.get("http://localhost:8000/project-info")
project_info = response.json()

# Get available models
response = requests.get("http://localhost:8000/models")
models = response.json()

# Make predictions
with open("data.csv", "rb") as f:
    files = {"csv_file": f}
    data = {"model_name": "random_forest_model"}
    response = requests.post("http://localhost:8000/validate", files=files, data=data)
    predictions = response.json()
```

### JavaScript with fetch
```javascript
// Get project information
const projectInfo = await fetch('http://localhost:8000/project-info')
  .then(response => response.json());

// Make predictions
const formData = new FormData();
formData.append('csv_file', csvFile);
formData.append('model_name', 'random_forest_model');

const predictions = await fetch('http://localhost:8000/validate', {
  method: 'POST',
  body: formData
}).then(response => response.json());
```

### cURL
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Get available models
curl -X GET "http://localhost:8000/models"

# Make predictions
curl -X POST "http://localhost:8000/validate" \
  -F "csv_file=@your_data.csv" \
  -F "model_name=your_model_name"
```

## Running the API

### Development Server
```bash
# Install dependencies
uv sync

# Run the development server
uv run uvicorn src.main:app --reload

# The API will be available at http://localhost:8000
```

### Production Server
```bash
# Run with production settings
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Testing

The API includes comprehensive unit and integration tests:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test categories
uv run pytest -m unit
uv run pytest -m integration
```

## Dependencies

### Core Dependencies
- **FastAPI** (>=0.104.0): Web framework
- **uvicorn** (>=0.24.0): ASGI server
- **pandas** (>=2.3.2): Data manipulation
- **scikit-learn** (>=1.7.2): Machine learning models
- **python-multipart** (>=0.0.6): File upload support

### Development Dependencies
- **pytest** (>=7.4.0): Testing framework
- **pytest-asyncio** (>=0.21.0): Async testing support
- **pytest-cov** (>=4.1.0): Coverage reporting
- **httpx** (>=0.25.0): HTTP client for testing

## Error Handling

The API implements comprehensive error handling:

- **Validation Errors (422)**: Invalid request parameters or missing fields
- **Client Errors (400)**: Invalid file formats, empty files, or malformed data
- **Not Found Errors (404)**: Non-existent models or resources
- **Server Errors (500)**: Internal processing errors, model loading failures

All errors return JSON responses with descriptive error messages to help with debugging and integration.

## Rate Limiting

Currently, no rate limiting is implemented. For production deployments, consider implementing rate limiting based on your requirements.

## Security Considerations

- The API currently does not implement authentication or authorization
- File uploads are limited to CSV format for security
- Model files are loaded from a controlled directory structure
- Input validation is performed on all endpoints

For production use, consider implementing:
- API key authentication
- Request size limits
- Input sanitization
- HTTPS encryption
- CORS configuration