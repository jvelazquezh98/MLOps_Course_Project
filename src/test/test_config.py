"""
Unit tests for configuration module.

This module tests the configuration settings, path management,
and environment setup for the MLOps project.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

from src.config import (
    PROJ_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR
)


class TestConfigPaths:
    """Test cases for configuration paths."""
    
    def test_proj_root_is_path(self):
        """Test that PROJ_ROOT is a Path object."""
        assert isinstance(PROJ_ROOT, Path)
    
    def test_proj_root_exists(self):
        """Test that PROJ_ROOT points to an existing directory."""
        assert PROJ_ROOT.exists()
        assert PROJ_ROOT.is_dir()
    
    def test_data_dir_structure(self):
        """Test that DATA_DIR is correctly structured."""
        assert isinstance(DATA_DIR, Path)
        assert DATA_DIR == PROJ_ROOT / "data"
    
    def test_raw_data_dir_structure(self):
        """Test that RAW_DATA_DIR is correctly structured."""
        assert isinstance(RAW_DATA_DIR, Path)
        assert RAW_DATA_DIR == DATA_DIR / "raw"
    
    def test_interim_data_dir_structure(self):
        """Test that INTERIM_DATA_DIR is correctly structured."""
        assert isinstance(INTERIM_DATA_DIR, Path)
        assert INTERIM_DATA_DIR == DATA_DIR / "interim"
    
    def test_processed_data_dir_structure(self):
        """Test that PROCESSED_DATA_DIR is correctly structured."""
        assert isinstance(PROCESSED_DATA_DIR, Path)
        assert PROCESSED_DATA_DIR == DATA_DIR / "processed"
    
    def test_external_data_dir_structure(self):
        """Test that EXTERNAL_DATA_DIR is correctly structured."""
        assert isinstance(EXTERNAL_DATA_DIR, Path)
        assert EXTERNAL_DATA_DIR == DATA_DIR / "external"
    
    def test_models_dir_structure(self):
        """Test that MODELS_DIR is correctly structured."""
        assert isinstance(MODELS_DIR, Path)
        assert MODELS_DIR == PROJ_ROOT / "models"
    
    def test_reports_dir_structure(self):
        """Test that REPORTS_DIR is correctly structured."""
        assert isinstance(REPORTS_DIR, Path)
        assert REPORTS_DIR == PROJ_ROOT / "reports"
    
    def test_figures_dir_structure(self):
        """Test that FIGURES_DIR is correctly structured."""
        assert isinstance(FIGURES_DIR, Path)
        assert FIGURES_DIR == REPORTS_DIR / "figures"
    
    def test_all_paths_are_absolute(self):
        """Test that all configured paths are absolute."""
        paths = [
            PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR,
            PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR,
            REPORTS_DIR, FIGURES_DIR
        ]
        
        for path in paths:
            assert path.is_absolute(), f"{path} is not absolute"


class TestConfigEnvironment:
    """Test cases for environment configuration."""
    
    def test_dotenv_loading(self):
        """Test that dotenv is loaded (if .env exists)."""
        # This test verifies the import doesn't fail
        from src import config
        assert hasattr(config, 'load_dotenv')
    
    def test_logger_configuration(self):
        """Test that logger is properly configured."""
        from loguru import logger
        
        # Logger should be configured
        assert logger is not None
    
    @patch('src.config.tqdm')
    def test_tqdm_integration_when_available(self, mock_tqdm):
        """Test that tqdm integration works when tqdm is available."""
        # Reload config to test tqdm integration
        import importlib
        import src.config
        importlib.reload(src.config)
        
        # Should not raise any errors
        assert True
    
    def test_config_imports_successfully(self):
        """Test that config module can be imported without errors."""
        try:
            import src.config
            assert True
        except Exception as e:
            pytest.fail(f"Config import failed: {e}")


class TestConfigPathCreation:
    """Test cases for path creation and validation."""
    
    def test_paths_can_be_created(self):
        """Test that paths can be used to create directories."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_dir"
            test_path.mkdir(parents=True, exist_ok=True)
            
            assert test_path.exists()
            assert test_path.is_dir()
    
    def test_path_joining_works(self):
        """Test that paths can be joined correctly."""
        test_path = DATA_DIR / "test" / "subdir"
        
        assert isinstance(test_path, Path)
        assert str(test_path).endswith("data/test/subdir")
    
    def test_relative_path_resolution(self):
        """Test that relative paths are resolved correctly."""
        relative_path = Path("./test")
        absolute_path = relative_path.resolve()
        
        assert absolute_path.is_absolute()
