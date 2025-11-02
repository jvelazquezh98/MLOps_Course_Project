# src/modeling/train.py
from __future__ import annotations

from pathlib import Path
import json
import os
import random
import sys
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

app = typer.Typer(help="Entrenamiento con MLflow (compatible con flujo de notebook)")

# ---------------------------------------------------------
# Utilidades
# ---------------------------------------------------------
def setup_logger(debug: bool) -> None:
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level} | {message}")

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

ALLOWED_PARAM_KEYS = {
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "random_state",
}

DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": 42,
}

# ---------------------------------------------------------
# Núcleo de entrenamiento (alineado al notebook)
# ---------------------------------------------------------
def train_model(data_path: Path, target: str, params: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(f"Leyendo dataset: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {data_path}")
    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("El dataset está vacío.")

    # Target automático desde 'shares' si no existe
    if target not in df.columns:
        if "shares" in df.columns:
            logger.warning(
                f"El target '{target}' no existe. Se creará automáticamente desde 'shares'."
            )
            threshold = 1400  # ajustable
            df[target] = (df["shares"] > threshold).astype(int)
            out_with_target = data_path.parent / f"{data_path.stem}_with_target.csv"
            df.to_csv(out_with_target, index=False)
            logger.info(
                f"Target '{target}' creado (threshold={threshold}). "
                f"Balance: {df[target].value_counts().to_dict()}. "
                f"Guardado en: {out_with_target}"
            )
        else:
            raise ValueError(
                f"Target '{target}' no existe y no hay columna 'shares' para generarlo."
            )

    # Separación X/y
    feature_df = df.drop(columns=[target])
    y = df[target].astype(int)

      
    # ------------------------------------------------------------------------
    # Si el target viene de 'shares', tumba 'shares' del conjunto de features
    if "shares" in feature_df.columns:
        logger.warning("Removiendo 'shares' de las features para evitar fuga de información")
        feature_df = feature_df.drop(columns=["shares"])

    # Elimina automáticamente columnas no numéricas
    non_numeric = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(f"Eliminando columnas no numéricas: {non_numeric}")
        feature_df = feature_df.drop(columns=non_numeric)

    X = feature_df


    logger.debug(f"Dimensiones X: {X.shape}, y: {y.shape}")

    # Split (igual que en notebook típico)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=int(params.get("random_state", 42)),
        stratify=y
    )

    # Modelo (hiperparámetros estilo notebook)
    model = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if params["max_depth"] in (None, "null") else int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=None if params["max_features"] in (None, "none") else params["max_features"],
        random_state=int(params["random_state"]),
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Métricas principales (notebook-friendly)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Artefacto local de métricas
    os.makedirs("reports/metrics", exist_ok=True)
    with open("reports/metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model": model,
        "metrics": metrics,
        "X_example": X.head(5)
    }

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
@app.command()
def main(
    data_path: Path = typer.Option(Path("data/processed/features.csv"), "--data-path", "-d", help="CSV de features"),
    target: str = typer.Option("popular", "--target", "-t", help="Columna objetivo"),
    params: str = typer.Option(
        "{}", "--params",
        help='Hiperparámetros en JSON. Ej: --params \'{"n_estimators":400,"max_depth":16,"random_state":123}\''
    ),
    tracking_uri: str = typer.Option("http://127.0.0.1:8080", help="MLflow Tracking URI"),
    experiment_name: str = typer.Option("MLOps_Course_Project", help="Nombre del experimento"),
    run_name: str = typer.Option("rf_train", help="Nombre del run"),
    register_model_name: Optional[str] = typer.Option(None, help="Nombre en Model Registry (opcional)"),
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Modo depuración"),
):
    setup_logger(debug)

    try:
        # Parseo y validación de hiperparámetros
        user_params = json.loads(params) if params else {}
        if not isinstance(user_params, dict):
            raise ValueError("--params debe ser un JSON con objeto dict")

        unknown = set(user_params.keys()) - ALLOWED_PARAM_KEYS
        if unknown:
            raise ValueError(
                f"Claves no permitidas en --params: {sorted(list(unknown))}. "
                f"Permitidas: {sorted(list(ALLOWED_PARAM_KEYS))}"
            )

        merged = {**DEFAULT_PARAMS, **user_params}
        set_seed(int(merged.get("random_state", 42)))

        # MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            # Log de parámetros y metadata mínima
            mlflow.log_params(merged)
            mlflow.log_params({"data_path": str(data_path), "target": target})

            # Entrenamiento
            result = train_model(data_path=data_path, target=target, params=merged)

            # Métricas
            mlflow.log_metrics(result["metrics"])

            # Modelo con firma
            signature = infer_signature(result["X_example"], result["model"].predict(result["X_example"]))
            mlflow.sklearn.log_model(
                sk_model=result["model"],
                artifact_path="model",
                signature=signature,
                registered_model_name=register_model_name
            )

        logger.info(f"Entrenamiento completado. Métricas: {json.dumps(result['metrics'], ensure_ascii=False)}")

    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
