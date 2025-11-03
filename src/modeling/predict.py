# src/modeling/predict.py
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import joblib
import mlflow

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="Script para realizar inferencias con el modelo entrenado")

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
    mlflow_run_id: str = typer.Option(None, "--mlflow-run-id", help="Run ID del modelo en MLflow (opcional)"),
):
    """
    Realiza inferencias sobre datos nuevos usando el modelo entrenado.
    """
    logger.info("Iniciando proceso de inferencia...")

    # -------------------------------------------------------------------------
    # 1️ Cargar modelo (local o desde MLflow)
    # -------------------------------------------------------------------------
    try:
        if mlflow_run_id:
            logger.info(f"Cargando modelo desde MLflow (run_id={mlflow_run_id})...")
            model = mlflow.sklearn.load_model(f"runs:/{mlflow_run_id}/model")
        else:
            logger.info(f"Cargando modelo local desde: {model_path}")
            model = joblib.load(model_path)
        logger.success("Modelo cargado correctamente.")
    except Exception as e:
        logger.error(f" Error al cargar el modelo: {e}")
        raise typer.Exit(code=1)

    # -------------------------------------------------------------------------
    # 2️ Cargar dataset de features
    # -------------------------------------------------------------------------
    if not features_path.exists():
        logger.error(f"No se encontró el archivo de features: {features_path}")
        raise typer.Exit(code=1)

    df = pd.read_csv(features_path)
    logger.info(f"Features cargadas correctamente: {df.shape[0]} filas, {df.shape[1]} columnas")

    # -------------------------------------------------------------------------
    # 3️Preprocesar (eliminar columnas que no se usan en predicción)
    # -------------------------------------------------------------------------
    cols_to_drop = [c for c in ["popular", "shares"] if c in df.columns]
    if cols_to_drop:
        logger.info(f"Eliminando columnas no necesarias para la inferencia: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        logger.warning(f"Eliminando columnas no numéricas: {non_numeric}")
        df = df.drop(columns=non_numeric)

    # -------------------------------------------------------------------------
    # 4️Realizar inferencia
    # -------------------------------------------------------------------------
    try:
        logger.info("Realizando predicciones...")
        preds = model.predict(df)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:, 1]
        else:
            probs = None

        df["prediction"] = preds
        if probs is not None:
            df["probability"] = probs

        logger.success("Predicciones generadas correctamente.")
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        raise typer.Exit(code=1)

    # -------------------------------------------------------------------------
    # 5️Guardar resultados
    # -------------------------------------------------------------------------
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(predictions_path, index=False)
    logger.success(f"Predicciones guardadas en: {predictions_path}")

    logger.success(" Proceso de inferencia completado con éxito.")


if _name_ == "_main_":
    app()