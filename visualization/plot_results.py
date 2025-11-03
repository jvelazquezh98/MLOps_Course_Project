"""
Script para visualización completa de resultados de modelos ML

Este módulo genera visualizaciones comprehensivas de los resultados de entrenamiento,
incluyendo matrices de confusión, curvas ROC, distribuciones de predicciones, y más.

Author: MLOps Course Project Team
Version: 1.0.0
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report
)

app = typer.Typer(help="Generación de visualizaciones de resultados de modelos")

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def setup_logger(debug: bool = False) -> None:
    """Configura el logger"""
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level} | {message}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None
) -> None:
    """Genera y guarda matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Matriz de confusión guardada en: {save_path}")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None
) -> float:
    """Genera curva ROC y retorna AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Curva ROC guardada en: {save_path}")
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Path] = None
) -> None:
    """Genera curva Precision-Recall"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Curva Precision-Recall guardada en: {save_path}")
    plt.close()


def plot_prediction_distribution(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    title: str = "Prediction Distribution",
    save_path: Optional[Path] = None
) -> None:
    """Genera distribución de predicciones por clase"""
    plt.figure(figsize=(10, 6))
    
    # Separar probabilidades por clase real
    proba_class_0 = y_proba[y_true == 0]
    proba_class_1 = y_proba[y_true == 1]
    
    plt.hist(proba_class_0, bins=50, alpha=0.5, label='Clase 0 (Real)', color='blue')
    plt.hist(proba_class_1, bins=50, alpha=0.5, label='Clase 1 (Real)', color='red')
    plt.xlabel('Probabilidad Predicha')
    plt.ylabel('Frecuencia')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribución de predicciones guardada en: {save_path}")
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None
) -> None:
    """Genera gráfico de importancia de features"""
    # Ordenar features por importancia
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Importancia de features guardada en: {save_path}")
    plt.close()


@app.command()
def generate_all_plots(
    predictions_path: Path = typer.Option(
        Path("data/processed/predictions.csv"),
        help="Ruta al archivo de predicciones"
    ),
    model_name: str = typer.Option("model", help="Nombre del modelo"),
    output_dir: Path = typer.Option(
        Path("reports/figures"),
        help="Directorio de salida para las figuras"
    ),
    debug: bool = typer.Option(False, "--debug", help="Modo debug")
) -> None:
    """
    Genera todas las visualizaciones de resultados del modelo
    """
    setup_logger(debug)
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar predicciones
    if not predictions_path.exists():
        logger.error(f"No se encontró el archivo: {predictions_path}")
        raise typer.Exit(code=1)
    
    logger.info(f"Cargando predicciones desde: {predictions_path}")
    df = pd.read_csv(predictions_path)
    
    # Validar columnas necesarias
    required_cols = ['prediction']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Faltan columnas requeridas: {required_cols}")
        raise typer.Exit(code=1)
    
    y_pred = df['prediction'].values
    
    # Buscar columna de valores reales
    y_true_col = None
    for col in ['popular', 'target', 'label', 'y_true']:
        if col in df.columns:
            y_true_col = col
            break
    
    if y_true_col is None:
        logger.warning("No se encontró columna de valores reales. Solo se generarán gráficos básicos.")
        y_true = None
    else:
        y_true = df[y_true_col].values
        logger.info(f"Usando '{y_true_col}' como valores reales")
    
    # Buscar probabilidades
    y_proba = df['probability'].values if 'probability' in df.columns else None
    
    # Generar visualizaciones
    logger.info("Generando visualizaciones...")
    
    if y_true is not None:
        # Matriz de confusión
        plot_confusion_matrix(
            y_true, y_pred,
            title=f"Matriz de Confusión - {model_name}",
            save_path=output_dir / f"confusion_matrix_{model_name}.png"
        )
        
        if y_proba is not None:
            # Curva ROC
            plot_roc_curve(
                y_true, y_proba,
                title=f"Curva ROC - {model_name}",
                save_path=output_dir / f"roc_curve_{model_name}.png"
            )
            
            # Curva Precision-Recall
            plot_precision_recall_curve(
                y_true, y_proba,
                title=f"Curva Precision-Recall - {model_name}",
                save_path=output_dir / f"precision_recall_{model_name}.png"
            )
            
            # Distribución de predicciones
            plot_prediction_distribution(
                y_proba, y_true,
                title=f"Distribución de Predicciones - {model_name}",
                save_path=output_dir / f"prediction_distribution_{model_name}.png"
            )
    
    logger.success(f"✅ Visualizaciones generadas en: {output_dir}")


if __name__ == "__main__":
    app()
