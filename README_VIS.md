# Guía de Uso de Scripts de Evaluación y Visualización

Este documento explica cómo usar los nuevos scripts creados para completar los puntos faltantes del proyecto.

## 📊 1. Visualización de Resultados (`src/visualization/plot_results.py`)

### Descripción
Genera visualizaciones comprehensivas de los resultados de modelos ML, incluyendo:
- Matrices de confusión
- Curvas ROC
- Curvas Precision-Recall
- Distribuciones de predicciones
- Importancia de features

### Uso Básico

```bash
# Generar todas las visualizaciones para un modelo
python -m src.visualization.plot_results generate-all-plots \
  --predictions-path data/processed/predictions.csv \
  --model-name "random_forest" \
  --output-dir reports/figures
```

### Ejemplos

```bash
# Con modo debug
python -m src.visualization.plot_results generate-all-plots \
  --predictions-path data/processed/predictions.csv \
  --model-name "xgboost" \
  --output-dir reports/figures/xgb \
  --debug

# Para modelo específico
python -m src.visualization.plot_results generate-all-plots \
  --predictions-path data/processed/rf_predictions.csv \
  --model-name "rf_v1" \
  --output-dir reports/figures/rf_v1
```

### Salidas Generadas
- `confusion_matrix_{model_name}.png` - Matriz de confusión
- `roc_curve_{model_name}.png` - Curva ROC con AUC
- `precision_recall_{model_name}.png` - Curva Precision-Recall
- `prediction_distribution_{model_name}.png` - Distribución de probabilidades

---

## 🔍 2. Comparación de Modelos (`src/evaluation/compare_models.py`)

### Descripción
Compara automáticamente múltiples modelos ML y genera:
- Tabla comparativa de métricas
- Gráficos de barras comparativos
- Gráficos radar
- Reporte de texto con rankings

### Uso Básico

```bash
# Comparar todos los modelos en el directorio de métricas
python -m src.evaluation.compare_models compare \
  --metrics-dir reports/metrics \
  --output-dir reports/comparison
```

### Ejemplos

```bash
# Con modo debug
python -m src.evaluation.compare_models compare \
  --metrics-dir reports/metrics \
  --output-dir reports/comparison \
  --debug

# Directorio personalizado
python -m src.evaluation.compare_models compare \
  --metrics-dir models/experiment_metrics \
  --output-dir reports/model_comparison_2024
```

### Salidas Generadas
- `model_comparison.csv` - Tabla CSV con todas las métricas
- `metrics_comparison_bar.png` - Gráfico de barras comparativo
- `metrics_comparison_radar.png` - Gráfico radar comparativo
- `comparison_report.txt` - Reporte detallado en texto

### Formato de Entrada
Los archivos JSON de métricas deben tener esta estructura:

```json
{
  "model": "RandomForest",
  "metrics": {
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.88,
    "f1": 0.85,
    "AUC_test_final": 0.90
  }
}
```

---

## 📝 3. Documentación de Notebooks (`src/documentation/notebook_documenter.py`)

### Descripción
Agrega documentación inline detallada a notebooks Jupyter de forma automatizada.

### Comandos Disponibles

#### 3.1 Agregar Encabezado

```bash
python -m src.documentation.notebook_documenter add-header \
  notebooks/01_eda.ipynb \
  --title "Análisis Exploratorio de Datos" \
  --author "Tu Nombre" \
  --objective "Explorar y entender el dataset" \
  --description "Análisis detallado de distribuciones y correlaciones"
```

#### 3.2 Generar Plantilla de Documentación

```bash
# Analiza el notebook y genera una plantilla JSON
python -m src.documentation.notebook_documenter generate-template \
  notebooks/01_eda.ipynb \
  --output-path notebooks/01_eda_doc_template.json
```

Esto genera un archivo JSON que puedes editar:

```json
{
  "notebook_name": "01_eda",
  "sections": [
    {
      "number": 1,
      "title": "Carga de Datos",
      "objective": "[COMPLETAR: Objetivo de esta sección]",
      "description": "[COMPLETAR: Descripción detallada]",
      "num_cells": 3
    }
  ]
}
```

#### 3.3 Aplicar Documentación

```bash
# Aplica la plantilla editada al notebook
python -m src.documentation.notebook_documenter apply-documentation \
  notebooks/01_eda.ipynb \
  notebooks/01_eda_doc_template.json \
  --output-path notebooks/01_eda_documented.ipynb
```

### Ejemplos Completos

```bash
# Workflow completo de documentación
# 1. Generar plantilla
python -m src.documentation.notebook_documenter generate-template \
  notebooks/02_preprocessing.ipynb

# 2. Editar el archivo JSON generado manualmente

# 3. Aplicar documentación
python -m src.documentation.notebook_documenter apply-documentation \
  notebooks/02_preprocessing.ipynb \
  notebooks/02_preprocessing_doc_template.json

# 4. Agregar encabezado final
python -m src.documentation.notebook_documenter add-header \
  notebooks/02_preprocessing_documented.ipynb \
  --title "Preprocesamiento de Datos" \
  --author "Data Team" \
  --objective "Limpiar y transformar datos para modelado"
```

---

## 🚀 Workflow Completo Recomendado

### Paso 1: Entrenar Modelos y Generar Métricas
```bash
# Ya tienes esto con tu script de entrenamiento
python -m src.modeling.train --data-path data/processed/features.csv
```

### Paso 2: Generar Visualizaciones
```bash
# Para cada modelo entrenado
python -m src.visualization.plot_results generate-all-plots \
  --predictions-path data/processed/predictions.csv \
  --model-name "rf_model" \
  --output-dir reports/figures
```

### Paso 3: Comparar Modelos
```bash
# Comparar todos los modelos
python -m src.evaluation.compare_models compare \
  --metrics-dir reports/metrics \
  --output-dir reports/comparison
```

### Paso 4: Documentar Notebooks
```bash
# Para cada notebook
python -m src.documentation.notebook_documenter add-header \
  notebooks/03_training.ipynb \
  --title "Entrenamiento de Modelos" \
  --author "ML Team" \
  --objective "Entrenar y evaluar modelos de clasificación"
```

---

## 📋 Requisitos

Asegúrate de tener instaladas las dependencias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn typer loguru tabulate
```

O con uv:

```bash
uv pip install pandas numpy matplotlib seaborn scikit-learn typer loguru tabulate
```

---

## 🎯 Puntos de la Matriz Completados

Con estos scripts completas:

✅ **Resultados (15 pts)**: Visualizaciones comprehensivas y presentación clara de métricas

✅ **Métodos y técnicas (20 pts)**: Comparación automatizada justifica selección de modelos

✅ **Documentación inline**: Notebooks con documentación detallada y estructurada

---

## 💡 Tips

1. **Ejecuta los scripts después de cada entrenamiento** para mantener visualizaciones actualizadas

2. **Usa el modo --debug** cuando algo no funcione para ver logs detallados

3. **Personaliza las plantillas** en `DOCUMENTATION_TEMPLATES` según tus necesidades

4. **Integra con MLflow** para tracking automático de visualizaciones

---

## 🐛 Troubleshooting

### Error: "No se encontraron métricas"
- Verifica que los archivos JSON estén en `reports/metrics/`
- Revisa que tengan la estructura correcta

### Error: "Columna 'popular' no encontrada"
- Asegúrate de que tu CSV de predicciones incluya la columna target
- O modifica el script para usar otra columna

### Gráficos no se generan
- Verifica que matplotlib esté instalado correctamente
- En servidores sin GUI, usa backend 'Agg': `matplotlib.use('Agg')`

---

¿Necesitas ayuda? Revisa los logs con `--debug` o consulta la documentación inline de cada script.
