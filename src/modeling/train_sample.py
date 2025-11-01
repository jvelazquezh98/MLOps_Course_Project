from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()
mlflow.set_tracking_uri(uri="http://0.0.0.0:8080")

def train_model():
    # Load the Iris dataset
    X, y = datasets.load_iris(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 8888,
    }

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
    }

    return lr, params, metrics, X_train, y_train


# add command to give a name to the run
@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    #features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    #labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = "../../models/",
    run_name: str = typer.Option(None, help="Nombre del run en MLflow"),
    model_name: str = typer.Option(None, help="Nombre del modelo en MLflow")
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    if run_name is None:
        run_name = "run_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    if model_name is None:
        model_name = "model_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        print(model_name)
        logger.info("Model name not provided. Using default name:", model_name)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    model_trained, params, metrics, X_train, y_train = train_model()
    # Create a new MLflow Experiment
    mlflow.set_experiment(run_name)

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("accuracy", metrics["accuracy"])

        # Infer the model signature
        signature = infer_signature(X_train, model_trained.predict(X_train))

        # Log the model, which inherits the parameters and metric
        model_info = mlflow.sklearn.log_model(
            sk_model=model_trained,
            name="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )

        # Set a tag that we can use to remind ourselves what this model was for
        mlflow.set_logged_model_tags(
            model_info.model_id, {"Training Info": "Basic LR model for iris data"}
        )

        # save the model locally
        mlflow.sklearn.save_model(
            sk_model=model_trained,
            path=f"./models/{model_name}",
            signature=signature,
            input_example=X_train,
        )
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
