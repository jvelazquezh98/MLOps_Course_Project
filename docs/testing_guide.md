# MLOps Project Testing Guide

## Quick Start

### Run All Tests
```bash
uv run pytest src/test/ -v
```

### Run Tests with Coverage Report
```bash
uv run pytest src/test/ --cov=src --cov-report=html --cov-report=term
```

### Run Specific Test Categories

#### Configuration Tests
```bash
uv run pytest src/test/test_config.py -v
```

#### Feature Engineering Tests
```bash
uv run pytest src/test/test_features.py -v
```

#### Model Training & Prediction Tests
```bash
uv run pytest src/test/test_modeling.py -v
```

#### Model Evaluation Tests
```bash
uv run pytest src/test/test_evaluation.py -v
```

#### Visualization Tests
```bash
uv run pytest src/test/test_visualization.py -v
```

#### Integration Tests
```bash
uv run pytest src/test/test_integration.py -v
```

#### REST API Tests
```bash
uv run pytest src/test/test_rest_api.py src/test/test_main.py -v
```

## Test Structure

```
src/test/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and configuration
├── test_config.py              # Configuration tests
├── test_features.py            # Feature engineering tests
├── test_modeling.py            # Model training/prediction tests
├── test_evaluation.py          # Model evaluation tests
├── test_visualization.py       # Visualization tests
├── test_integration.py         # End-to-end integration tests
├── test_rest_api.py           # REST API unit tests
└── test_main.py               # FastAPI integration tests
```

## Test Categories

### Unit Tests
Test individual functions and components in isolation.

**Examples:**
- `test_config.py` - Path configuration
- `test_features.py` - Feature transformations
- `test_modeling.py` - Model training logic
- `test_evaluation.py` - Metrics extraction

### Integration Tests
Test complete workflows and component interactions.

**Examples:**
- `test_integration.py` - Full MLOps pipeline
- `test_main.py` - API with all endpoints

## Writing New Tests

### Test Naming Convention
```python
def test_<component>_<scenario>_<expected_result>():
    """
    Brief description of what is being tested.
    """
    # Arrange
    # Act
    # Assert
```

### Example Test
```python
def test_model_training_with_valid_data_returns_metrics():
    """Test that model training returns expected metrics."""
    # Arrange
    data_path = create_sample_data()
    params = DEFAULT_PARAMS
    
    # Act
    result = train_model(data_path, target='popular', params=params)
    
    # Assert
    assert 'metrics' in result
    assert 'accuracy' in result['metrics']
    assert 0 <= result['metrics']['accuracy'] <= 1
```

## Using Fixtures

### Available Fixtures (from conftest.py)

```python
def test_with_temp_models_dir(temp_models_dir):
    """Use temporary models directory."""
    model_path = temp_models_dir / "test_model.pkl"
    # ... test code

def test_with_sample_data(sample_dataframe):
    """Use sample DataFrame."""
    assert len(sample_dataframe) > 0
    # ... test code

def test_with_mock_model(mock_model):
    """Use mock ML model."""
    predictions = mock_model.predict([[1, 2, 3]])
    # ... test code
```

## Test Markers

### Run Only Unit Tests
```bash
uv run pytest src/test/ -m unit -v
```

### Run Only Integration Tests
```bash
uv run pytest src/test/ -m integration -v
```

### Run Async Tests
```bash
uv run pytest src/test/ -m asyncio -v
```

## Debugging Tests

### Run with Detailed Output
```bash
uv run pytest src/test/test_modeling.py -vv
```

### Run with Print Statements
```bash
uv run pytest src/test/test_modeling.py -s
```

### Run Specific Test
```bash
uv run pytest src/test/test_modeling.py::TestModelTraining::test_train_model_returns_metrics -v
```

### Stop on First Failure
```bash
uv run pytest src/test/ -x
```

### Run Last Failed Tests
```bash
uv run pytest src/test/ --lf
```

## Coverage Analysis

### Generate HTML Coverage Report
```bash
uv run pytest src/test/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Show Missing Lines
```bash
uv run pytest src/test/ --cov=src --cov-report=term-missing
```

### Coverage for Specific Module
```bash
uv run pytest src/test/ --cov=src.modeling --cov-report=term
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run tests
        run: uv run pytest src/test/ -v --cov=src
```

## Common Issues

### Issue: Tests Fail Due to Missing Dependencies
**Solution:**
```bash
uv sync
```

### Issue: Tests Fail Due to Path Issues
**Solution:** Run tests from project root directory
```bash
cd /path/to/MLOps_Course_Project
uv run pytest src/test/ -v
```

### Issue: Async Tests Fail
**Solution:** Ensure pytest-asyncio is installed
```bash
uv add pytest-asyncio
```

### Issue: Model Files Interfere with Tests
**Solution:** Tests use temporary directories, but if needed:
```bash
# Clean up test artifacts
rm -rf .pytest_cache
rm -rf htmlcov
```

## Performance Testing

### Measure Test Execution Time
```bash
uv run pytest src/test/ --durations=10
```

### Run Tests in Parallel
```bash
uv add pytest-xdist
uv run pytest src/test/ -n auto
```

## Best Practices

1. **Keep Tests Fast**: Unit tests should run in milliseconds
2. **Use Fixtures**: Reuse common setup code
3. **Test Edge Cases**: Test boundary conditions and error cases
4. **Mock External Dependencies**: Don't rely on external services
5. **Clean Up Resources**: Use context managers and temp files
6. **Descriptive Names**: Test names should explain what they test
7. **One Concept Per Test**: Each test should verify one thing
8. **Arrange-Act-Assert**: Follow the AAA pattern

## Test Data

### Creating Test Data
```python
import tempfile
import pandas as pd
from pathlib import Path

def create_test_csv():
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        return Path(f.name)
```

### Cleaning Up Test Data
```python
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    # Create test files in tmpdir
    # Files are automatically cleaned up when context exits
    pass
```

## Troubleshooting

### View Test Collection
```bash
uv run pytest src/test/ --collect-only
```

### Debug Test Discovery Issues
```bash
uv run pytest src/test/ -v --tb=short
```

### Check Test Configuration
```bash
cat pytest.ini
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
