# Suite de Pruebas del Proyecto MLOps

## Descripción General

Suite de pruebas comprehensiva para el Proyecto del Curso MLOps, cubriendo pruebas unitarias y de integración para todo el pipeline de machine learning.

## Inicio Rápido

```bash
# Ejecutar todas las pruebas
pytest -q

# Ejecutar con salida detallada
pytest -v

# Ejecutar con cobertura
pytest --cov=src --cov-report=html
```

## Estructura de Pruebas

```
src/test/
├── __init__.py                    # Inicialización del paquete de pruebas
├── conftest.py                    # Fixtures compartidos y configuración
├── test_config.py                 # Pruebas de configuración y rutas (18 pruebas)
├── test_features.py               # Pruebas de ingeniería de features (11 pruebas)
├── test_modeling.py               # Pruebas de entrenamiento/predicción (17 pruebas)
├── test_evaluation.py             # Pruebas de evaluación de modelos (18 pruebas)
├── test_visualization.py          # Pruebas de visualización (24 pruebas)
├── test_integration.py            # Pruebas de integración E2E (17 pruebas)
├── test_plot_results.py           # Pruebas de generación de gráficos (42 pruebas)
├── test_notebook_documenter.py    # Pruebas de documentación (39 pruebas)
├── test_compare_models.py         # Pruebas de comparación de modelos (43 pruebas)
├── test_rest_api.py              # Pruebas unitarias de REST API
└── test_main.py                  # Pruebas de integración FastAPI
```

## Categorías de Pruebas

### Pruebas Unitarias (215+ pruebas)
Prueban componentes individuales de forma aislada:
- **Configuración** (`test_config.py`) - Gestión de rutas, configuración del entorno
- **Ingeniería de Features** (`test_features.py`) - Preprocesamiento, limpieza de datos
- **Modelado** (`test_modeling.py`) - Entrenamiento, predicción, persistencia
- **Evaluación** (`test_evaluation.py`) - Cálculo de métricas, comparación
- **Visualización** (`test_visualization.py`) - Generación de gráficos, preparación de datos
- **Generación de Gráficos** (`test_plot_results.py`) - Matrices de confusión, curvas ROC
- **Documentación** (`test_notebook_documenter.py`) - Notebooks Jupyter, análisis de código
- **Comparación de Modelos** (`test_compare_models.py`) - Comparación multi-modelo, reportes

### Pruebas de Integración (28 pruebas)
Prueban flujos de trabajo completos:
- **Pipeline de Datos** - Datos crudos → Procesamiento → Features
- **Pipeline de Modelos** - Entrenamiento → Evaluación → Persistencia
- **Flujo MLOps** - Validación del pipeline extremo a extremo
- **Integración API** - Carga de modelos y predicción a través de API
- **Monitoreo** - Seguimiento de rendimiento y detección de drift

## Ejecutar Pruebas

### Todas las Pruebas
```bash
# Modo silencioso (recomendado)
pytest -q

# Modo detallado
pytest -v

# Detener en el primer fallo
pytest -x
```

### Archivos de Prueba Específicos
```bash
# Pruebas de ingeniería de features
pytest src/test/test_features.py -v

# Pruebas de entrenamiento de modelos
pytest src/test/test_modeling.py -v

# Pruebas de integración
pytest src/test/test_integration.py -v

# Pruebas de generación de gráficos
pytest src/test/test_plot_results.py -v

# Pruebas de documentación de notebooks
pytest src/test/test_notebook_documenter.py -v

# Pruebas de comparación de modelos
pytest src/test/test_compare_models.py -v
```

### Prueba Específica
```bash
pytest src/test/test_modeling.py::TestModelTraining::test_train_model_returns_metrics -v
```

### Con Cobertura
```bash
# Reporte en terminal
pytest --cov=src --cov-report=term

# Reporte HTML
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Fixtures de Prueba

Fixtures disponibles desde `conftest.py`:

- `temp_models_dir` - Directorio temporal para archivos de modelos
- `sample_csv_data` - Cadena de datos CSV de ejemplo
- `sample_dataframe` - DataFrame de pandas de ejemplo
- `mock_model` - Modelo ML simulado para pruebas
- `sample_metadata` - Estructura de metadatos de ejemplo
- `create_test_model_file` - Ayudante para crear modelos de prueba
- `create_test_csv_file` - Ayudante para crear archivos CSV de prueba

## Marcadores de Prueba

```bash
# Ejecutar solo pruebas unitarias
pytest -m unit

# Ejecutar solo pruebas de integración
pytest -m integration

# Ejecutar solo pruebas asíncronas
pytest -m asyncio
```

## Escribir Nuevas Pruebas

### Convención de Nombres de Pruebas
```python
def test_<componente>_<escenario>_<resultado_esperado>():
    """Breve descripción de lo que se está probando."""
    # Arrange (Preparar)
    # Act (Actuar)
    # Assert (Afirmar)
```

### Ejemplo
```python
def test_model_training_with_valid_data_returns_metrics():
    """Prueba que el entrenamiento del modelo retorna métricas esperadas."""
    # Arrange (Preparar)
    data_path = create_sample_data()
    params = DEFAULT_PARAMS
    
    # Act (Actuar)
    result = train_model(data_path, target='popular', params=params)
    
    # Assert (Afirmar)
    assert 'metrics' in result
    assert 0 <= result['metrics']['accuracy'] <= 1
```

## Resultados de Pruebas

Estado actual:
- **Total de Pruebas**: 263
- **Aprobadas**: ~258 (98%)
- **Pruebas Nuevas**: 124 (esta sesión)
- **Tiempo de Ejecución**: ~35 segundos

## Cobertura

### Componentes Probados
- ✅ Ingesta y preprocesamiento de datos
- ✅ Ingeniería de features
- ✅ Entrenamiento y evaluación de modelos
- ✅ Persistencia y carga de modelos
- ✅ Cálculo y comparación de métricas
- ✅ Generación de visualizaciones
- ✅ Generación de gráficos (matrices, curvas ROC, etc.)
- ✅ Documentación de notebooks
- ✅ Comparación de modelos
- ✅ Integración con API
- ✅ Flujos de trabajo extremo a extremo

### Cobertura por Módulo
```
test_config.py:              17/18 (94%)
test_features.py:            11/11 (100%)
test_modeling.py:            17/17 (100%)
test_evaluation.py:          18/18 (100%)
test_visualization.py:       24/24 (100%)
test_integration.py:         17/17 (100%)
test_plot_results.py:        41/42 (97.6%)
test_notebook_documenter.py: 39/39 (100%)
test_compare_models.py:      43/43 (100%)
```

## Depuración

### Salida Detallada
```bash
pytest -vv
```

### Mostrar Declaraciones Print
```bash
pytest -s
```

### Ejecutar Últimas Pruebas Fallidas
```bash
pytest --lf
```

### Mostrar Duración de Pruebas
```bash
pytest --durations=10
```

## Integración CI/CD

Las pruebas están diseñadas para pipelines CI/CD:
- Ejecución rápida (< 40 segundos)
- Aisladas (usan directorios temporales)
- Determinísticas (semillas aleatorias fijas)
- Sin dependencias externas

### Ejemplo de GitHub Actions
```yaml
- name: Ejecutar pruebas
  run: |
    pip install uv
    uv sync
    uv run pytest src/test/ -v --cov=src
```

## Documentación

Para información más detallada, consulta:
- `../../TEST_SUMMARY.md` - Resumen completo de pruebas
- `../../docs/testing_guide.md` - Guía detallada de pruebas
- `../../VALIDATION_REPORT.md` - Validación de requisitos
- `../../QUICK_TEST_REFERENCE.md` - Tarjeta de referencia rápida
- `../../FINAL_TEST_IMPLEMENTATION.md` - Implementación final completa
- `../../COMPLETE_TEST_SUMMARY.md` - Resumen completo actualizado

## Mejores Prácticas

1. **Mantén las pruebas rápidas** - Las pruebas unitarias deben ejecutarse en milisegundos
2. **Usa fixtures** - Reutiliza código de configuración común
3. **Prueba casos extremos** - Incluye condiciones límite
4. **Simula dependencias externas** - No dependas de servicios externos
5. **Limpia recursos** - Usa context managers y archivos temporales
6. **Nombres descriptivos** - Los nombres de pruebas deben explicar qué prueban
7. **Un concepto por prueba** - Cada prueba debe verificar una cosa
8. **Sigue el patrón AAA** - Arrange (Preparar), Act (Actuar), Assert (Afirmar)

## Solución de Problemas

### Las pruebas fallan por dependencias faltantes
```bash
uv sync
```

### Las pruebas fallan por problemas de rutas
Ejecuta desde la raíz del proyecto:
```bash
cd /ruta/a/MLOps_Course_Project
pytest src/test/ -v
```

### Limpiar artefactos de prueba
```bash
rm -rf .pytest_cache htmlcov
```

## Contribuir

Al agregar nuevas pruebas:
1. Sigue la estructura de pruebas existente
2. Usa nombres descriptivos para las pruebas
3. Agrega docstrings
4. Usa fixtures apropiados
5. Limpia recursos
6. Actualiza este README si es necesario

## Soporte

Para preguntas o problemas:
1. Revisa los archivos de documentación
2. Revisa las pruebas existentes como ejemplos
3. Consulta `docs/testing_guide.md`
