"""
Script para comparación automatizada de métricas entre modelos

Este módulo compara múltiples modelos ML, genera reportes comparativos
y visualizaciones para facilitar la selección del mejor modelo.

Author: MLOps Course Project Team
Version: 1.0.0
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from tabulate import tabulate

app = typer.Typer(help="Comparación automatizada de modelos ML")

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def setup_logger(debug: bool = False) -> None:
    """Configura el logger"""
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level} | {message}")


def load_metrics_from_json(metrics_path: Path) -> Dict[str, Any]:
    """Carga métricas desde archivo JSON"""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando {metrics_path}: {e}")
        return {}


def extract_key_metrics(metrics_data: Dict[str, Any]) -> Dict[str, float]:
    """Extrae métricas clave del JSON de métricas"""
    key_metrics = {}
    
    # Intentar extraer métricas de diferentes estructuras
    if 'metrics' in metrics_data:
        metrics = metrics_data['metrics']
        
        # Métricas directas
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'AUC_test_final']:
            if metric in metrics:
                key_metrics[metric] = metrics[metric]
        
        # Métricas del classification_report
        if 'classification_report' in metrics:
            report = metrics['classification_report']
            if 'weighted_avg' in report:
                weighted = report['weighted_avg']
                key_metrics['weighted_precision'] = weighted.get('precision', 0)
                key_metrics['weighted_recall'] = weighted.get('recall', 0)
                key_metrics['weighted_f1'] = weighted.get('f1_score', 0)
    
    return key_metrics


def create_comparison_dataframe(metrics_dir: Path) -> pd.DataFrame:
    """Crea DataFrame con comparación de todos los modelos"""
    models_data = []
    
    # Buscar todos los archivos JSON en el directorio
    for metrics_file in metrics_dir.glob("*.json"):
        logger.info(f"Procesando: {metrics_file.name}")
        
        metrics_data = load_metrics_from_json(metrics_file)
        if not metrics_data:
            continue
        
        # Extraer información del modelo
        model_info = {
            'model_file': metrics_file.stem,
            'model_name': metrics_data.get('model', 'Unknown')
        }
        
        # Extraer métricas clave
        key_metrics = extract_key_metrics(metrics_data)
        model_info.update(key_metrics)
        
        models_data.append(model_info)
    
    if not models_data:
        logger.warning("No se encontraron métricas válidas")
        return pd.DataFrame()
    
    df = pd.DataFrame(models_data)
    logger.success(f"Comparación creada con {len(df)} modelos")
    
    return df


def plot_metrics_comparison(
    df: pd.DataFrame,
    metrics: List[str],
    title: str = "Comparación de Métricas entre Modelos",
    save_path: Optional[Path] = None
) -> None:
    """Genera gráfico de barras comparando métricas"""
    # Filtrar solo las métricas que existen
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        logger.warning("No hay métricas disponibles para graficar")
        return
    
    # Preparar datos
    df_plot = df[['model_name'] + available_metrics].set_index('model_name')
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.legend(title='Métricas', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico guardado en: {save_path}")
    plt.close()


def plot_radar_chart(
    df: pd.DataFrame,
    metrics: List[str],
    title: str = "Comparación Radar de Modelos",
    save_path: Optional[Path] = None
) -> None:
    """Genera gráfico radar comparando modelos"""
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics or len(df) == 0:
        logger.warning("Datos insuficientes para gráfico radar")
        return
    
    # Configurar ángulos
    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Graficar cada modelo
    for idx, row in df.iterrows():
        values = [row[m] for m in available_metrics]
        values += values[:1]  # Cerrar el círculo
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model_name'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico radar guardado en: {save_path}")
    plt.close()


def generate_comparison_report(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """Genera reporte de comparación en formato texto"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE DE COMPARACIÓN DE MODELOS\n")
        f.write("=" * 80 + "\n\n")
        
        # Tabla de métricas
        f.write("## Tabla Comparativa de Métricas\n\n")
        table = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
        f.write(table + "\n\n")
        
        # Mejor modelo por métrica
        f.write("## Mejor Modelo por Métrica\n\n")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            best_idx = df[col].idxmax()
            best_model = df.loc[best_idx, 'model_name']
            best_value = df.loc[best_idx, col]
            f.write(f"- **{col}**: {best_model} ({best_value:.4f})\n")
        
        f.write("\n")
        
        # Ranking general
        f.write("## Ranking General (Promedio de Métricas)\n\n")
        df_copy = df.copy()
        df_copy['avg_score'] = df_copy[numeric_cols].mean(axis=1)
        df_sorted = df_copy.sort_values('avg_score', ascending=False)
        
        for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
            f.write(f"{idx}. {row['model_name']}: {row['avg_score']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.success(f"Reporte generado en: {output_path}")


@app.command()
def compare(
    metrics_dir: Path = typer.Option(
        Path("reports/metrics"),
        help="Directorio con archivos JSON de métricas"
    ),
    output_dir: Path = typer.Option(
        Path("reports/comparison"),
        help="Directorio de salida para comparaciones"
    ),
    debug: bool = typer.Option(False, "--debug", help="Modo debug")
) -> None:
    """
    Compara todos los modelos y genera reportes y visualizaciones
    """
    setup_logger(debug)
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar que existe el directorio de métricas
    if not metrics_dir.exists():
        logger.error(f"No existe el directorio: {metrics_dir}")
        raise typer.Exit(code=1)
    
    logger.info(f"Buscando métricas en: {metrics_dir}")
    
    # Crear DataFrame de comparación
    df = create_comparison_dataframe(metrics_dir)
    
    if df.empty:
        logger.error("No se encontraron métricas para comparar")
        raise typer.Exit(code=1)
    
    # Guardar CSV de comparación
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparación guardada en: {csv_path}")
    
    # Métricas principales a comparar
    main_metrics = ['accuracy', 'precision', 'recall', 'f1', 'AUC_test_final']
    
    # Generar gráfico de barras
    plot_metrics_comparison(
        df,
        main_metrics,
        title="Comparación de Métricas entre Modelos",
        save_path=output_dir / "metrics_comparison_bar.png"
    )
    
    # Generar gráfico radar
    plot_radar_chart(
        df,
        main_metrics,
        title="Comparación Radar de Modelos",
        save_path=output_dir / "metrics_comparison_radar.png"
    )
    
    # Generar reporte de texto
    generate_comparison_report(
        df,
        output_path=output_dir / "comparison_report.txt"
    )
    
    # Mostrar resumen en consola
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE COMPARACIÓN")
    logger.info("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df['avg_score'] = df[numeric_cols].mean(axis=1)
    best_model = df.loc[df['avg_score'].idxmax()]
    
    logger.info(f"\n🏆 Mejor modelo: {best_model['model_name']}")
    logger.info(f"   Score promedio: {best_model['avg_score']:.4f}\n")
    
    for col in main_metrics:
        if col in df.columns:
            logger.info(f"   {col}: {best_model[col]:.4f}")
    
    logger.success(f"\n✅ Comparación completada. Resultados en: {output_dir}")


if __name__ == "__main__":
    app()
