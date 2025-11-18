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


## Etapas del Proyecto

### Fase 1: Análisis y Preparación de Datos (2 semanas)

#### Semana 1 (29 Sept – 5 Oct)
**Objetivo:**  
Generar 2 conjuntos de datos (divididos en train/test) para entrenar los modelos.

**Tareas de Análisis**
- Exploratory Data Analysis (EDA).
- Generación de datasets train/test.  
    - Roles: **Software Engineer**, **Data Engineer**

**Tareas de Gestión**
- Creación de scripts de transformación (migración de notebooks a scripts `.py` para automatización).  
- Implementación de un pipeline de transformación y versionado de datos.  
    - Rol: **DevOps**

---

#### Semana 2 (6 – 12 Oct)
**Objetivo:**  
Utilizar los datasets para entrenar modelos, generar archivos `.pkl` y pipelines de entrenamiento e inferencia.

**Tareas de Análisis**
- Entrenamiento de modelos con diferentes algoritmos.
- Documentación y análisis de resultados.  
    - Roles: **Data Scientist**, **ML Engineer**

**Tareas de Gestión**
- Creación de scripts de entrenamiento e inferencia (migración de notebooks a scripts `.py`).  
- Generación de métricas finales para evaluación de modelos.  
    - Rol: **DevOps**

---

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

El proyecto incluye una API REST para servir predicciones del modelo:

```bash
# Iniciar el servidor
uvicorn src.main:app --reload

# Acceder a la documentación
http://localhost:8000/docs
```

### Endpoints Disponibles

- `GET /` - Información de la API
- `GET /health` - Health check
- `GET /project-info` - Información del proyecto
- `GET /datasets` - Información de datasets
- `GET /models` - Modelos disponibles
- `POST /validate` - Realizar predicciones

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
