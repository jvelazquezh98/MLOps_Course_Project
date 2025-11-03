"""
Script para agregar documentación inline detallada a notebooks

Este módulo analiza notebooks Jupyter y agrega documentación inline
comprehensiva siguiendo mejores prácticas de MLOps.

Author: MLOps Course Project Team
Version: 1.0.0
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import typer
from loguru import logger

app = typer.Typer(help="Documentación automatizada de notebooks")


def setup_logger(debug: bool = False) -> None:
    """Configura el logger"""
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level} | {message}")


# Plantillas de documentación
DOCUMENTATION_TEMPLATES = {
    "header": """# {title}

**Autor:** {author}  
**Fecha:** {date}  
**Objetivo:** {objective}

## Descripción
{description}

## Contenido
{toc}

---
""",
    
    "section": """## {section_number}. {section_title}

**Objetivo:** {objective}

**Descripción:**
{description}
""",
    
    "code_cell": """### Código: {code_title}

**Propósito:** {purpose}

**Inputs:**
{inputs}

**Outputs:**
{outputs}

**Notas:**
{notes}
""",
    
    "analysis": """### Análisis de Resultados

**Hallazgos clave:**
{findings}

**Interpretación:**
{interpretation}

**Próximos pasos:**
{next_steps}
""",
    
    "conclusion": """## Conclusiones

### Resumen
{summary}

### Logros
{achievements}

### Limitaciones
{limitations}

### Recomendaciones
{recommendations}
"""
}


def load_notebook(notebook_path: Path) -> Dict[str, Any]:
    """Carga un notebook Jupyter"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando notebook: {e}")
        raise


def save_notebook(notebook: Dict[str, Any], output_path: Path) -> None:
    """Guarda un notebook Jupyter"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        logger.success(f"Notebook guardado en: {output_path}")
    except Exception as e:
        logger.error(f"Error guardando notebook: {e}")
        raise


def create_markdown_cell(content: str) -> Dict[str, Any]:
    """Crea una celda markdown"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }


def analyze_code_cell(cell: Dict[str, Any]) -> Dict[str, str]:
    """Analiza una celda de código y extrae información"""
    source = ''.join(cell.get('source', []))
    
    analysis = {
        "has_imports": "import " in source,
        "has_visualization": any(x in source for x in ["plt.", "sns.", "plot"]),
        "has_model": any(x in source for x in ["fit(", "predict(", "train"]),
        "has_data_loading": any(x in source for x in ["read_csv", "load_data", "pd.read"]),
        "has_preprocessing": any(x in source for x in ["fillna", "dropna", "transform", "scale"]),
    }
    
    return analysis


def add_section_documentation(
    notebook: Dict[str, Any],
    section_configs: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Agrega documentación de secciones al notebook"""
    new_cells = []
    cell_idx = 0
    
    for section in section_configs:
        # Agregar celda de documentación de sección
        section_doc = DOCUMENTATION_TEMPLATES["section"].format(
            section_number=section.get("number", ""),
            section_title=section.get("title", ""),
            objective=section.get("objective", ""),
            description=section.get("description", "")
        )
        new_cells.append(create_markdown_cell(section_doc))
        
        # Agregar celdas originales de esta sección
        num_cells = section.get("num_cells", 1)
        for _ in range(num_cells):
            if cell_idx < len(notebook['cells']):
                new_cells.append(notebook['cells'][cell_idx])
                cell_idx += 1
    
    # Agregar celdas restantes
    while cell_idx < len(notebook['cells']):
        new_cells.append(notebook['cells'][cell_idx])
        cell_idx += 1
    
    notebook['cells'] = new_cells
    return notebook


def generate_documentation_template(notebook_path: Path) -> Dict[str, Any]:
    """Genera plantilla de documentación para un notebook"""
    notebook = load_notebook(notebook_path)
    
    # Analizar estructura del notebook
    sections = []
    current_section = {"cells": [], "number": 1}
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            # Detectar encabezados de sección
            if source.startswith('##'):
                if current_section['cells']:
                    sections.append(current_section)
                current_section = {
                    "cells": [cell],
                    "number": len(sections) + 1,
                    "title": source.replace('#', '').strip()
                }
            else:
                current_section['cells'].append(cell)
        else:
            current_section['cells'].append(cell)
    
    if current_section['cells']:
        sections.append(current_section)
    
    # Generar configuración de documentación
    doc_config = {
        "notebook_name": notebook_path.stem,
        "sections": []
    }
    
    for section in sections:
        section_config = {
            "number": section['number'],
            "title": section.get('title', f"Sección {section['number']}"),
            "objective": "[COMPLETAR: Objetivo de esta sección]",
            "description": "[COMPLETAR: Descripción detallada]",
            "num_cells": len(section['cells'])
        }
        doc_config["sections"].append(section_config)
    
    return doc_config


@app.command()
def generate_template(
    notebook_path: Path = typer.Argument(..., help="Ruta al notebook"),
    output_path: Path = typer.Option(
        None,
        help="Ruta de salida para la plantilla (default: notebook_name_doc_template.json)"
    ),
    debug: bool = typer.Option(False, "--debug", help="Modo debug")
) -> None:
    """
    Genera una plantilla de documentación para un notebook existente
    """
    setup_logger(debug)
    
    if not notebook_path.exists():
        logger.error(f"No existe el notebook: {notebook_path}")
        raise typer.Exit(code=1)
    
    logger.info(f"Analizando notebook: {notebook_path}")
    
    # Generar plantilla
    doc_config = generate_documentation_template(notebook_path)
    
    # Determinar ruta de salida
    if output_path is None:
        output_path = notebook_path.parent / f"{notebook_path.stem}_doc_template.json"
    
    # Guardar plantilla
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(doc_config, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Plantilla generada en: {output_path}")
    logger.info("Edita la plantilla y luego ejecuta 'apply-documentation'")


@app.command()
def apply_documentation(
    notebook_path: Path = typer.Argument(..., help="Ruta al notebook"),
    template_path: Path = typer.Argument(..., help="Ruta a la plantilla de documentación"),
    output_path: Path = typer.Option(
        None,
        help="Ruta de salida (default: notebook_name_documented.ipynb)"
    ),
    debug: bool = typer.Option(False, "--debug", help="Modo debug")
) -> None:
    """
    Aplica documentación a un notebook usando una plantilla
    """
    setup_logger(debug)
    
    if not notebook_path.exists():
        logger.error(f"No existe el notebook: {notebook_path}")
        raise typer.Exit(code=1)
    
    if not template_path.exists():
        logger.error(f"No existe la plantilla: {template_path}")
        raise typer.Exit(code=1)
    
    logger.info(f"Aplicando documentación a: {notebook_path}")
    
    # Cargar notebook y plantilla
    notebook = load_notebook(notebook_path)
    
    with open(template_path, 'r', encoding='utf-8') as f:
        doc_config = json.load(f)
    
    # Aplicar documentación
    documented_notebook = add_section_documentation(
        notebook,
        doc_config.get('sections', [])
    )
    
    # Determinar ruta de salida
    if output_path is None:
        output_path = notebook_path.parent / f"{notebook_path.stem}_documented.ipynb"
    
    # Guardar notebook documentado
    save_notebook(documented_notebook, output_path)
    
    logger.success(f"✅ Notebook documentado guardado en: {output_path}")


@app.command()
def add_header(
    notebook_path: Path = typer.Argument(..., help="Ruta al notebook"),
    title: str = typer.Option(..., "--title", help="Título del notebook"),
    author: str = typer.Option("MLOps Team", "--author", help="Autor"),
    objective: str = typer.Option(..., "--objective", help="Objetivo principal"),
    description: str = typer.Option("", "--description", help="Descripción detallada"),
    output_path: Path = typer.Option(None, help="Ruta de salida"),
    debug: bool = typer.Option(False, "--debug", help="Modo debug")
) -> None:
    """
    Agrega un encabezado documentado al inicio del notebook
    """
    setup_logger(debug)
    
    if not notebook_path.exists():
        logger.error(f"No existe el notebook: {notebook_path}")
        raise typer.Exit(code=1)
    
    # Cargar notebook
    notebook = load_notebook(notebook_path)
    
    # Crear encabezado
    from datetime import datetime
    header = DOCUMENTATION_TEMPLATES["header"].format(
        title=title,
        author=author,
        date=datetime.now().strftime("%Y-%m-%d"),
        objective=objective,
        description=description or "Análisis y procesamiento de datos para el proyecto MLOps",
        toc="[Generado automáticamente]"
    )
    
    # Insertar al inicio
    header_cell = create_markdown_cell(header)
    notebook['cells'].insert(0, header_cell)
    
    # Determinar ruta de salida
    if output_path is None:
        output_path = notebook_path
    
    # Guardar
    save_notebook(notebook, output_path)
    
    logger.success(f"✅ Encabezado agregado a: {output_path}")


if __name__ == "__main__":
    app()
