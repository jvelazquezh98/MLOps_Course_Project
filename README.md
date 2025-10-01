# MLOps Course Project

Este proyecto tiene como objetivo demostrar las capacidades de los estudiantes en las distintas etapas del desarrollo de proyectos de Machine Learning, desde la manipulación y preparación de datos hasta la construcción y evaluación de modelos.

## Estructura del Proyecto

```
MLOps_Course_Project/
├─ .venv/
├─ data/
│ ├─ interim/
│ ├─ processed/
│ └─ raw/
├─ docs/
├─ notebooks/
├─ reports/
├─ src/
├─ .dvcignore
├─ .env
├─ .gitignore
├─ .python-version
├─ main.py
├─ pyproject.toml
├─ README.md
└─ uv.lock
```

- `data/raw/`: Contiene los datos originales sin modificar.
- `data/interim/`: Datos intermedios generados durante la limpieza o transformación.
- `data/processed/`: Datos finales listos para el modelado.
- `notebooks/`: Notebooks de análisis exploratorio y experimentación.
- `reports/`: Informes y presentaciones ejecutivas.
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
