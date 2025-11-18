# MLOps_Course_Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Este proyecto tiene como objetivo demostrar las capacidades de los estudiantes en las distintas etapas del desarrollo de proyectos de Machine Learning, desde la manipulación y preparación de datos hasta la construcción y evaluación de modelos.

## Estructura del Proyecto

```
├── LICENSE            <- Licencia open-source
├── Makefile           <- Makefile con comandos generales
├── README.md          <- Top-level README para desarrolladores del proyecto
├── data
│   ├── interim        <- Data en proceso intermedio de transformación
│   ├── processed      <- Datasets finales
│   └── raw            <- Datasets originales
│
├── docs               <- Default proyecto mkdocs; consultar www.mkdocs.org para mas detalles
│
├── models             <- Modelos entrenados y serializados
│
├── notebooks          <- Jupyter notebooks.
│
├── pyproject.toml     <- Configuración del proyecto con metadata de paqueterías.
│
├── references         <- Diccionarios de datos, manuales, materiales explicativos.
│
├── reports            <- Análisis generados como HTML, PDF, LaTeX, etc.
│   └── figures        <- Gráficos generados.
│
├── setup.cfg          <- File de configuración para flake8
│
└── src   <- Source code para el proyecto
    │
    ├── __init__.py             <- Inicializa módulo de python
    │
    ├── config.py               <- Guarda variables y configuraciones útiles
    │
    ├── dataset.py              <- Scripts para descargar o generar datos
    │
    ├── features.py             <- Código para generar features para modelar
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Código para hacer inferencia con modelos entrenados
    │   └── train.py            <- Código para entrenar modelos
    │
    ├── plots.py                <- Código para crear visualizacioes
    │
    ├── rest_api.py             <- REST API endpoints para el modelo
    │
    ├── main.py                 <- FastAPI application entry point
    │
    └── test                    <- Test suite para el proyecto
        ├── __init__.py
        ├── conftest.py         <- Fixtures compartidos
        ├── test_config.py      <- Tests de configuración
        ├── test_features.py    <- Tests de feature engineering
        ├── test_modeling.py    <- Tests de entrenamiento/predicción
        ├── test_evaluation.py  <- Tests de evaluación de modelos
        ├── test_visualization.py <- Tests de visualización
        ├── test_integration.py <- Tests de integración E2E
        ├── test_rest_api.py    <- Tests de API
        └── test_main.py        <- Tests de aplicación FastAPI
```

- `data/raw/`: Contiene los datos originales sin modificar.
- `data/interim/`: Datos intermedios generados durante la limpieza o transformación.
- `data/processed/`: Datos finales listos para el modelado.
- `notebooks/`: Notebooks de análisis exploratorio y experimentación.
- `reports/figures/`: Imagenes generadas durante el proyecto.
- `reports/metrics/`: Métricas de los modelos entrenados.
- `src/`: Código fuente del proyecto.
- `.dvc/`: Archivos y configuraciones de DVC.
- `.venv/`: Entorno virtual del proyecto.

---

## Requerimientos del proyecto

1. **Nomenclatura de Datasets**  
   Para identificar fácilmente las versiones y etapas de los datos, los archivos deben seguir la siguiente convención:

   `<dataset>_<fase>_<fecha>.csv`

   - `<dataset>`: nombre corto del dataset (ej. `ventas`, `clientes`).  
    - `<fase>`: puede ser `raw`, `interim`, `processed`.  
    - `<fecha>`: formato `YYYYMMDD` (ej. `20250930`).  

    **Ejemplo:**  
    - clientes_raw_20250930.csv
    - ventas_processed_20251002.csv 
2. **Nomenclatura de Archivos**
    - Nomenclatura de notebooks:  
        - `<número>_<tema>_<autor>.ipynb`
    - `<número>`: indica el orden del flujo de trabajo (ej. `01`, `02`).
    - `<tema>`: tema o propósito principal del notebook (`eda`, `preprocessing`, `training`, etc.).
    - `<autor>`: iniciales o nombre corto del responsable.

    **Ejemplos:**  
    01_eda_jv.ipynb
    02_preprocessing_cd.ipynb
    03_training_rf_mg.ipynb

3. **Figuras de análisis exploratorios**  
    Todas las figuras generadas durante el EDA deben guardarse en la carpeta: `reports/figures/`

    Con nomenclatura: `<tema>_<nombre_dataset>_.png`

    **Ejemplo:**
    - tendencias_ventas_20251001.png
    - distribucion_clientes_raw_20250930.png


4. **Nomenclatura de modelos**  
    Los modelos entrenados y guardados en formato `.pkl` deben seguir esta convención: 
    
    `<algoritmo>_<dataset>_<version>.pkl`

    - `<algoritmo>`: abreviatura del modelo (`rf` = Random Forest, `xgb` = XGBoost, `lr` = Logistic Regression, etc.).  
    - `<dataset>`: dataset con el que fue entrenado.  
    - `<version>`: versión incremental `v1`, `v2`, etc.  

    **Ejemplo:**
    - rf_clientes_v1.pkl
    - xgb_ventas_v3.pkl

---

## Instructivo: Descarga de Datos Versionados

Para acceder a los datasets versionados con DVC, sigue estos pasos:

1. Cargar el archivo `.env` con las credenciales de AWS S3 en carpeta raíz del proyecto.  

El archivo .env debe contener algo como:

```bash
AWS_ACCESS_KEY_ID=<tu_access_key_id>
AWS_SECRET_ACCESS_KEY=<tu_secret_access_key>
AWS_DEFAULT_REGION=<tu_region>
```

2. Ejecutar el comando para descargar los datos versionados:

```bash
uv run dvc pull
```

Esto traerá los datos en sus diferentes versiones y los ubicará en las carpetas correspondientes dentro de data/.

## Pruebas

### Ejecutar Pruebas

El proyecto incluye un suite completo de pruebas unitarias e integración:

```bash
# Ejecutar todas las pruebas
pytest -q

# Ejecutar con cobertura
pytest --cov=src --cov-report=html

# Ejecutar pruebas específicas
pytest src/test/test_integration.py -v
```

### Cobertura de Pruebas

- ✅ **107 pruebas implementadas**
- ✅ **99% de éxito en pruebas nuevas**
- ✅ **Cobertura completa del pipeline MLOps**

**Componentes Validados:**
- Preprocesamiento de datos (11 pruebas)
- Entrenamiento de modelos (17 pruebas)
- Evaluación y métricas (18 pruebas)
- Visualizaciones (24 pruebas)
- Integración extremo a extremo (17 pruebas)
- API REST (14 pruebas)

### Documentación de Pruebas

Para más información sobre las pruebas:
- `TEST_SUMMARY.md` - Resumen completo de pruebas
- `docs/testing_guide.md` - Guía detallada para desarrolladores
- `src/test/README.md` - Documentación del directorio de pruebas
- `TEST_EXECUTION_REPORT.md` - Reporte de ejecución
- `VALIDATION_REPORT.md` - Validación de requisitos

---

## API REST

El proyecto incluye una API REST para servir predicciones del modelo y gestionar el ciclo de vida de los modelos ML.

### Iniciar el Servidor

```bash
# Iniciar el servidor
uvicorn src.main:app --reload

# Iniciar con configuración personalizada
uv run -m src.main --host 0.0.0.0 --port 8000

### Endpoints Disponibles

#### **General**

##### `GET /` - Información de la API
Endpoint raíz que proporciona información básica sobre la API.

**Respuesta:**
```json
{
  "message": "MLOps Course Project API",
  "version": "0.1.0",
  "status": "running"
}
```

**Ejemplo:**
```bash
curl http://localhost:8000/
```

---

##### `GET /health` - Health Check
Verifica que el servicio de API esté funcionando correctamente.

**Respuesta:**
```json
{
  "status": "healthy"
}
```

**Ejemplo:**
```bash
curl http://localhost:8000/health
```

---

#### **Proyecto**

##### `GET /project-info` - Información del Proyecto
Obtiene información detallada sobre el proyecto MLOps, incluyendo configuración, metadata de ingeniería de datos y features implementadas.

**Respuesta:**
```json
{
  "project_name": "mlops-course-project",
  "description": "MLOps Course Project demonstrating ML pipeline capabilities",
  "python_version": "3.13",
  "data_engineering_metadata": {...},
  "features": [
    "Data Version Control (DVC)",
    "ML Pipeline",
    "Model Training",
    "Data Processing"
  ]
}
```

**Ejemplo:**
```bash
curl http://localhost:8000/project-info
```

---

#### **Datos**

##### `GET /datasets` - Información de Datasets
Obtiene información sobre los datasets disponibles y la estructura del pipeline de datos.

**Respuesta:**
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

**Ejemplo:**
```bash
curl http://localhost:8000/datasets
```

---

#### **Modelos**

##### `GET /models` - Modelos Disponibles
Lista todos los modelos entrenados disponibles para hacer predicciones.

**Respuesta:**
```json
{
  "available_models": [
    {
      "name": "rf_model_20251114_181516",
      "filename": "model.pkl",
      "path": "models/rf_model_20251114_181516"
    }
  ],
  "total_models": 1
}
```

**Ejemplo:**
```bash
curl http://localhost:8000/models
```

---

#### **Predicción**

##### `POST /validate` - Realizar Predicciones
Valida datos y realiza predicciones usando un modelo entrenado. Acepta un archivo CSV y retorna las predicciones junto con metadata del proceso.

**Parámetros:**
- `csv_file` (file, requerido): Archivo CSV con los datos para predicción
- `model_name` (string, requerido): Nombre del modelo a usar (nombre de la carpeta del modelo)

**Respuesta:**
```json
{
  "status": "success",
  "model_used": "rf_model_20251114_181516",
  "input_shape": [100, 60],
  "features_used": 58,
  "predictions_count": 100,
  "predictions": [0, 1, 1, 0, ...],
  "original_columns": ["url", "feature1", "feature2", ...],
  "features_columns": ["feature1", "feature2", ...],
  "message": "Successfully made 100 predictions using model 'rf_model_20251114_181516'"
}
```

**Ejemplo:**
```bash
# Usando curl
curl -X POST "http://localhost:8000/validate" \
  -F "csv_file=@data/processed/features.csv" \
  -F "model_name=rf_model_20251114_181516"

# Guardar predicciones en archivo
curl -X POST "http://localhost:8000/validate" \
  -F "csv_file=@data/processed/features.csv" \
  -F "model_name=rf_model_20251114_181516" \
  -o predictions.json
```

**Notas:**
- El archivo CSV puede contener columnas no numéricas (como 'url'), que serán eliminadas automáticamente
- El modelo utiliza solo las features numéricas con las que fue entrenado
- Si el modelo tiene `feature_names_in_`, se validará que el CSV contenga todas las features requeridas

---

#### **Entrenamiento**

##### `POST /train` - Entrenar Nuevo Modelo
Entrena un nuevo modelo Random Forest con los parámetros especificados. Utiliza MLflow para tracking de experimentos y guarda el modelo localmente.

**Parámetros:**
- `data_path` (string, opcional): Ruta al archivo CSV de entrenamiento
- `target` (string, opcional): Nombre de la columna objetivo
- `params` (string JSON, opcional): Hiperparámetros del modelo en formato JSON
- `ignore_drift` (boolean, opcional): Ignorar detección de drift en datos (default: false)

**Parámetros permitidos en `params`:**
```json
{
  "n_estimators": 300,
  "max_depth": 12,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "sqrt",
  "random_state": 42
}
```

**Respuesta:**
```json
{
  "message": "Model training test completed successfully.",
  "metrics": {
    "accuracy": 0.8543,
    "precision": 0.8321,
    "recall": 0.8765,
    "f1": 0.8538
  }
}
```

**Ejemplo:**
```bash
# Entrenamiento básico
curl -X POST "http://localhost:8000/train" \
  -F "data_path=data/processed/features_with_target.csv" \
  -F "target=is_popular"

# Entrenamiento con hiperparámetros personalizados
curl -X POST "http://localhost:8000/train" \
  -F "data_path=data/processed/features_with_target.csv" \
  -F "target=is_popular" \
  -F 'params={"n_estimators": 200, "max_depth": 15, "random_state": 42}'

# Entrenamiento ignorando drift
curl -X POST "http://localhost:8000/train" \
  -F "data_path=data/processed/features_with_target.csv" \
  -F "target=is_popular" \
  -F "ignore_drift=true"
```

**Notas:**
- Si no se proporciona el target y existe la columna 'shares', se creará automáticamente como clasificación binaria (shares > 1400)
- El endpoint detecta drift en los datos y puede fallar si se detecta (a menos que `ignore_drift=true`)
- Los modelos se guardan localmente en `./models/` y se registran en MLflow

---

### Códigos de Estado HTTP

- `200` - Operación exitosa
- `400` - Error en los parámetros de entrada o datos inválidos
- `404` - Modelo o archivo no encontrado
- `500` - Error interno del servidor

---

### Manejo de Errores

Todos los endpoints retornan errores en formato JSON:

```json
{
  "detail": "Descripción del error"
}
```

**Ejemplos de errores comunes:**

```bash
# Modelo no encontrado
{
  "detail": "Model 'modelo_inexistente' not found. Available models can be checked at /models endpoint"
}

# Archivo CSV inválido
{
  "detail": "File must be a CSV file"
}

# Parámetros de entrenamiento inválidos
{
  "detail": "Unknown parameters: ['param_invalido']. Allowed: ['n_estimators', 'max_depth', ...]"
}

# Drift detectado
{
  "detail": "Data drift detected based on the provided metrics. To continue training, please address the drift issue or run training with param 'ignore_drift=True'."
}
```

---

## Desarrollo

### Configuración del Entorno

```bash
# Instalar dependencias
uv sync

# Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Ejecutar Scripts

```bash
# Feature engineering
uv run python -m src.features

# Entrenar modelo
uv run python -m src.modeling.train --data-path data/processed/features.csv

# Hacer predicciones
uv run python -m src.modeling.predict --model-id model_name
```

### Calidad de Código

```bash
# Ejecutar pruebas
pytest -v

# Verificar cobertura
pytest --cov=src --cov-report=term

# Linting (si configurado)
flake8 src/
```
