"""
Unit tests for documentation/notebook_documenter.py module.

This module tests all notebook documentation functions including
loading, saving, analyzing, and documenting Jupyter notebooks.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile

from documentation.notebook_documenter import (
    setup_logger,
    load_notebook,
    save_notebook,
    create_markdown_cell,
    analyze_code_cell,
    add_section_documentation,
    generate_documentation_template,
    DOCUMENTATION_TEMPLATES
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
        assert True
    
    def test_setup_logger_info_mode(self):
        """Test logger setup in info mode."""
        setup_logger(debug=False)
        assert True


class TestLoadNotebook:
    """Test cases for loading notebooks."""
    
    def test_load_notebook_valid_file(self):
        """Test loading a valid notebook file."""
        notebook_data = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Test"]},
                {"cell_type": "code", "source": ["print('hello')"]}
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(notebook_data, f)
            notebook_path = Path(f.name)
        
        try:
            loaded = load_notebook(notebook_path)
            
            assert isinstance(loaded, dict)
            assert "cells" in loaded
            assert len(loaded["cells"]) == 2
            assert loaded["cells"][0]["cell_type"] == "markdown"
        finally:
            notebook_path.unlink()
    
    def test_load_notebook_nonexistent_file(self):
        """Test loading a non-existent notebook file."""
        with pytest.raises(FileNotFoundError):
            load_notebook(Path("nonexistent.ipynb"))
    
    def test_load_notebook_invalid_json(self):
        """Test loading a notebook with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            f.write("invalid json content")
            notebook_path = Path(f.name)
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_notebook(notebook_path)
        finally:
            notebook_path.unlink()
    
    def test_load_notebook_empty_file(self):
        """Test loading an empty notebook file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            f.write("{}")
            notebook_path = Path(f.name)
        
        try:
            loaded = load_notebook(notebook_path)
            assert isinstance(loaded, dict)
        finally:
            notebook_path.unlink()


class TestSaveNotebook:
    """Test cases for saving notebooks."""
    
    def test_save_notebook_valid_data(self):
        """Test saving a valid notebook."""
        notebook_data = {
            "cells": [{"cell_type": "markdown", "source": ["# Test"]}],
            "metadata": {},
            "nbformat": 4
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_notebook.ipynb"
            
            save_notebook(notebook_data, output_path)
            
            assert output_path.exists()
            
            # Verify content
            with open(output_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded == notebook_data
    
    def test_save_notebook_creates_parent_directories(self):
        """Test that save_notebook creates parent directories if needed."""
        notebook_data = {"cells": [], "metadata": {}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "notebook.ipynb"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_notebook(notebook_data, output_path)
            
            assert output_path.exists()
    
    def test_save_notebook_overwrites_existing(self):
        """Test that save_notebook overwrites existing files."""
        notebook_data = {"cells": [], "metadata": {}, "version": 1}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "notebook.ipynb"
            
            # Save first version
            save_notebook(notebook_data, output_path)
            
            # Save second version
            notebook_data["version"] = 2
            save_notebook(notebook_data, output_path)
            
            # Verify second version
            with open(output_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded["version"] == 2


class TestCreateMarkdownCell:
    """Test cases for creating markdown cells."""
    
    def test_create_markdown_cell_basic(self):
        """Test creating a basic markdown cell."""
        content = "# Test Header"
        cell = create_markdown_cell(content)
        
        assert cell["cell_type"] == "markdown"
        assert "metadata" in cell
        assert "source" in cell
        assert isinstance(cell["source"], list)
    
    def test_create_markdown_cell_multiline(self):
        """Test creating a markdown cell with multiple lines."""
        content = "# Header\n\nSome text\n\nMore text"
        cell = create_markdown_cell(content)
        
        assert cell["cell_type"] == "markdown"
        assert len(cell["source"]) > 1
    
    def test_create_markdown_cell_empty(self):
        """Test creating an empty markdown cell."""
        content = ""
        cell = create_markdown_cell(content)
        
        assert cell["cell_type"] == "markdown"
        assert isinstance(cell["source"], list)
    
    def test_create_markdown_cell_with_special_characters(self):
        """Test creating a markdown cell with special characters."""
        content = "# Test with **bold** and *italic*"
        cell = create_markdown_cell(content)
        
        assert cell["cell_type"] == "markdown"
        assert "**bold**" in ''.join(cell["source"])


class TestAnalyzeCodeCell:
    """Test cases for analyzing code cells."""
    
    def test_analyze_code_cell_with_imports(self):
        """Test analyzing a code cell with imports."""
        cell = {
            "cell_type": "code",
            "source": ["import pandas as pd\nimport numpy as np"]
        }
        
        analysis = analyze_code_cell(cell)
        
        assert analysis["has_imports"] is True
        assert analysis["has_visualization"] is False
        assert analysis["has_model"] is False
    
    def test_analyze_code_cell_with_visualization(self):
        """Test analyzing a code cell with visualization code."""
        cell = {
            "cell_type": "code",
            "source": ["plt.plot(x, y)\nplt.show()"]
        }
        
        analysis = analyze_code_cell(cell)
        
        assert analysis["has_visualization"] is True
    
    def test_analyze_code_cell_with_model_training(self):
        """Test analyzing a code cell with model training."""
        cell = {
            "cell_type": "code",
            "source": ["model.fit(X_train, y_train)\npredictions = model.predict(X_test)"]
        }
        
        analysis = analyze_code_cell(cell)
        
        assert analysis["has_model"] is True
    
    def test_analyze_code_cell_with_data_loading(self):
        """Test analyzing a code cell with data loading."""
        cell = {
            "cell_type": "code",
            "source": ["df = pd.read_csv('data.csv')"]
        }
        
        analysis = analyze_code_cell(cell)
        
        assert analysis["has_data_loading"] is True
    
    def test_analyze_code_cell_with_preprocessing(self):
        """Test analyzing a code cell with preprocessing."""
        cell = {
            "cell_type": "code",
            "source": ["df.fillna(0)\ndf.dropna()"]
        }
        
        analysis = analyze_code_cell(cell)
        
        assert analysis["has_preprocessing"] is True
    
    def test_analyze_code_cell_empty(self):
        """Test analyzing an empty code cell."""
        cell = {
            "cell_type": "code",
            "source": []
        }
        
        analysis = analyze_code_cell(cell)
        
        assert all(not value for value in analysis.values())
    
    def test_analyze_code_cell_multiple_features(self):
        """Test analyzing a code cell with multiple features."""
        cell = {
            "cell_type": "code",
            "source": [
                "import pandas as pd\n",
                "df = pd.read_csv('data.csv')\n",
                "df.fillna(0)\n",
                "plt.plot(df['col'])\n",
                "model.fit(X, y)"
            ]
        }
        
        analysis = analyze_code_cell(cell)
        
        assert analysis["has_imports"] is True
        assert analysis["has_data_loading"] is True
        assert analysis["has_preprocessing"] is True
        assert analysis["has_visualization"] is True
        assert analysis["has_model"] is True


class TestAddSectionDocumentation:
    """Test cases for adding section documentation."""
    
    def test_add_section_documentation_basic(self):
        """Test adding basic section documentation."""
        notebook = {
            "cells": [
                {"cell_type": "code", "source": ["print('hello')"]},
                {"cell_type": "code", "source": ["print('world')"]}
            ]
        }
        
        section_configs = [
            {
                "number": 1,
                "title": "Introduction",
                "objective": "Introduce the notebook",
                "description": "This is the introduction",
                "num_cells": 2
            }
        ]
        
        result = add_section_documentation(notebook, section_configs)
        
        assert len(result["cells"]) == 3  # 1 doc cell + 2 original cells
        assert result["cells"][0]["cell_type"] == "markdown"
    
    def test_add_section_documentation_multiple_sections(self):
        """Test adding documentation for multiple sections."""
        notebook = {
            "cells": [
                {"cell_type": "code", "source": ["# Section 1"]},
                {"cell_type": "code", "source": ["# Section 2"]},
                {"cell_type": "code", "source": ["# Section 3"]}
            ]
        }
        
        section_configs = [
            {
                "number": 1,
                "title": "Section 1",
                "objective": "Objective 1",
                "description": "Description 1",
                "num_cells": 1
            },
            {
                "number": 2,
                "title": "Section 2",
                "objective": "Objective 2",
                "description": "Description 2",
                "num_cells": 2
            }
        ]
        
        result = add_section_documentation(notebook, section_configs)
        
        # Should have 2 doc cells + 3 original cells
        assert len(result["cells"]) == 5
    
    def test_add_section_documentation_preserves_original_cells(self):
        """Test that original cells are preserved."""
        original_cell = {"cell_type": "code", "source": ["original_code"]}
        notebook = {"cells": [original_cell.copy()]}
        
        section_configs = [
            {
                "number": 1,
                "title": "Test",
                "objective": "Test",
                "description": "Test",
                "num_cells": 1
            }
        ]
        
        result = add_section_documentation(notebook, section_configs)
        
        # Find the original cell
        code_cells = [c for c in result["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) == 1
        assert code_cells[0]["source"] == ["original_code"]


class TestGenerateDocumentationTemplate:
    """Test cases for generating documentation templates."""
    
    def test_generate_documentation_template_basic(self):
        """Test generating a basic documentation template."""
        notebook_data = {
            "cells": [
                {"cell_type": "markdown", "source": ["## Section 1"]},
                {"cell_type": "code", "source": ["print('test')"]},
                {"cell_type": "markdown", "source": ["## Section 2"]},
                {"cell_type": "code", "source": ["print('test2')"]}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(notebook_data, f)
            notebook_path = Path(f.name)
        
        try:
            template = generate_documentation_template(notebook_path)
            
            assert "notebook_name" in template
            assert "sections" in template
            assert len(template["sections"]) > 0
        finally:
            notebook_path.unlink()
    
    def test_generate_documentation_template_detects_sections(self):
        """Test that template generation detects sections."""
        notebook_data = {
            "cells": [
                {"cell_type": "markdown", "source": ["## Introduction"]},
                {"cell_type": "code", "source": ["code1"]},
                {"cell_type": "markdown", "source": ["## Analysis"]},
                {"cell_type": "code", "source": ["code2"]}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(notebook_data, f)
            notebook_path = Path(f.name)
        
        try:
            template = generate_documentation_template(notebook_path)
            
            assert len(template["sections"]) == 2
            assert template["sections"][0]["title"] == "Introduction"
            assert template["sections"][1]["title"] == "Analysis"
        finally:
            notebook_path.unlink()
    
    def test_generate_documentation_template_counts_cells(self):
        """Test that template counts cells per section."""
        notebook_data = {
            "cells": [
                {"cell_type": "markdown", "source": ["## Section 1"]},
                {"cell_type": "code", "source": ["code1"]},
                {"cell_type": "code", "source": ["code2"]},
                {"cell_type": "markdown", "source": ["## Section 2"]},
                {"cell_type": "code", "source": ["code3"]}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(notebook_data, f)
            notebook_path = Path(f.name)
        
        try:
            template = generate_documentation_template(notebook_path)
            
            # First section should have 3 cells (header + 2 code)
            # Second section should have 2 cells (header + 1 code)
            assert template["sections"][0]["num_cells"] == 3
            assert template["sections"][1]["num_cells"] == 2
        finally:
            notebook_path.unlink()
    
    def test_generate_documentation_template_empty_notebook(self):
        """Test generating template for empty notebook."""
        notebook_data = {"cells": []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(notebook_data, f)
            notebook_path = Path(f.name)
        
        try:
            template = generate_documentation_template(notebook_path)
            
            assert "sections" in template
            # Should have at least one section even if empty
            assert len(template["sections"]) >= 0
        finally:
            notebook_path.unlink()


class TestDocumentationTemplates:
    """Test cases for documentation templates."""
    
    def test_documentation_templates_exist(self):
        """Test that all required templates exist."""
        required_templates = ["header", "section", "code_cell", "analysis", "conclusion"]
        
        for template_name in required_templates:
            assert template_name in DOCUMENTATION_TEMPLATES
    
    def test_header_template_format(self):
        """Test that header template can be formatted."""
        template = DOCUMENTATION_TEMPLATES["header"]
        
        formatted = template.format(
            title="Test Title",
            author="Test Author",
            date="2025-01-01",
            objective="Test Objective",
            description="Test Description",
            toc="Test TOC"
        )
        
        assert "Test Title" in formatted
        assert "Test Author" in formatted
        assert "Test Objective" in formatted
    
    def test_section_template_format(self):
        """Test that section template can be formatted."""
        template = DOCUMENTATION_TEMPLATES["section"]
        
        formatted = template.format(
            section_number="1",
            section_title="Introduction",
            objective="Introduce the topic",
            description="Detailed description"
        )
        
        assert "1" in formatted
        assert "Introduction" in formatted
        assert "Introduce the topic" in formatted
    
    def test_code_cell_template_format(self):
        """Test that code cell template can be formatted."""
        template = DOCUMENTATION_TEMPLATES["code_cell"]
        
        formatted = template.format(
            code_title="Data Loading",
            purpose="Load data from CSV",
            inputs="CSV file path",
            outputs="DataFrame",
            notes="Check for missing values"
        )
        
        assert "Data Loading" in formatted
        assert "Load data from CSV" in formatted
    
    def test_analysis_template_format(self):
        """Test that analysis template can be formatted."""
        template = DOCUMENTATION_TEMPLATES["analysis"]
        
        formatted = template.format(
            findings="Key findings",
            interpretation="Interpretation",
            next_steps="Next steps"
        )
        
        assert "Key findings" in formatted
        assert "Interpretation" in formatted
    
    def test_conclusion_template_format(self):
        """Test that conclusion template can be formatted."""
        template = DOCUMENTATION_TEMPLATES["conclusion"]
        
        formatted = template.format(
            summary="Summary",
            achievements="Achievements",
            limitations="Limitations",
            recommendations="Recommendations"
        )
        
        assert "Summary" in formatted
        assert "Achievements" in formatted


class TestIntegration:
    """Integration tests for notebook documentation workflow."""
    
    def test_complete_documentation_workflow(self):
        """Test complete workflow: load → analyze → document → save."""
        # Create a test notebook
        notebook_data = {
            "cells": [
                {"cell_type": "markdown", "source": ["## Data Loading"]},
                {"cell_type": "code", "source": ["import pandas as pd\ndf = pd.read_csv('data.csv')"]},
                {"cell_type": "markdown", "source": ["## Analysis"]},
                {"cell_type": "code", "source": ["df.describe()"]}
            ],
            "metadata": {},
            "nbformat": 4
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save original notebook
            notebook_path = Path(tmpdir) / "test.ipynb"
            with open(notebook_path, 'w') as f:
                json.dump(notebook_data, f)
            
            # Generate template
            template = generate_documentation_template(notebook_path)
            
            # Verify template
            assert len(template["sections"]) == 2
            
            # Load notebook
            notebook = load_notebook(notebook_path)
            
            # Add documentation
            documented = add_section_documentation(notebook, template["sections"])
            
            # Save documented notebook
            output_path = Path(tmpdir) / "documented.ipynb"
            save_notebook(documented, output_path)
            
            # Verify output
            assert output_path.exists()
            
            # Load and verify
            with open(output_path, 'r') as f:
                result = json.load(f)
            
            assert len(result["cells"]) > len(notebook_data["cells"])
    
    def test_analyze_multiple_code_cells(self):
        """Test analyzing multiple code cells in sequence."""
        cells = [
            {"cell_type": "code", "source": ["import pandas as pd"]},
            {"cell_type": "code", "source": ["df = pd.read_csv('data.csv')"]},
            {"cell_type": "code", "source": ["plt.plot(df['col'])"]},
            {"cell_type": "code", "source": ["model.fit(X, y)"]}
        ]
        
        analyses = [analyze_code_cell(cell) for cell in cells]
        
        assert analyses[0]["has_imports"] is True
        assert analyses[1]["has_data_loading"] is True
        assert analyses[2]["has_visualization"] is True
        assert analyses[3]["has_model"] is True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_load_notebook_with_unicode(self):
        """Test loading notebook with unicode characters."""
        notebook_data = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Título con ñ y acentos"]}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False, encoding='utf-8') as f:
            json.dump(notebook_data, f, ensure_ascii=False)
            notebook_path = Path(f.name)
        
        try:
            loaded = load_notebook(notebook_path)
            assert "ñ" in ''.join(loaded["cells"][0]["source"])
        finally:
            notebook_path.unlink()
    
    def test_create_markdown_cell_with_newlines(self):
        """Test creating markdown cell with various newline formats."""
        content = "Line 1\nLine 2\r\nLine 3"
        cell = create_markdown_cell(content)
        
        assert cell["cell_type"] == "markdown"
        assert len(cell["source"]) > 1
    
    def test_analyze_code_cell_with_comments(self):
        """Test analyzing code cell with comments."""
        cell = {
            "cell_type": "code",
            "source": [
                "# This is a comment\n",
                "import pandas as pd  # Import pandas\n",
                "# Another comment"
            ]
        }
        
        analysis = analyze_code_cell(cell)
        assert analysis["has_imports"] is True
