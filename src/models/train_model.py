import numpy as np
import pandas as pd
import argparse
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import yaml
import logging
from mlflow.tracking import MlflowClient
import platform
import sklearn


# ----------------------------------
# Configure logging
# ----------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None, help="Mlflow tracking URI")
    return parser.parse_args()

# -------------------------------
# Load model from config
# -------------------------------
def get_model_instance(name, params):
    model_map = {
        "LogisticRegression": LogisticRegression,
        "DecisionTree": DecisionTreeClassifier,
        "SVC": SVC
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)


# ---------------------------------
# Main logic
# ----------------------------------
def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg["name"])

    # Load data
    data = pd.read_csv(args.data)
    target = model_cfg["target_variable"]

    # Use all features except the target variable
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get model
    model = get_model_instance(model_cfg["best_model"], model_cfg["parameters"])

    # Start Mlflow run
    with mlflow.start_run(run_name="final_training"):
        logger.info(f"Training model: {model_cfg['best_model']}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log params and metrics
        mlflow.log_params(model_cfg["parameters"])
        mlflow.log_metrics({"accuracy": accuracy,
                            "precision": precision,
                            "recall": recall})
        
        # Log and register model
        mlflow.sklearn.log_model(model, name="tunned model", input_example=X_train)
        model_name = model_cfg["name"]
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/tunned_model"

        logger.info("Registering model to Mlflow Model Registery...")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass

        model_version = client.create_model_version(
            name = model_name,
            source = model_uri,
            run_id = mlflow.active_run().info.run_id
        )

        # Transition model to "Staging"
        client.transition_model_version_stage(
            name = model_name,
            version = model_version.version,
            stage = "Staging"
        )

        # Add a human-readable description
        description = (
            f"Model for predicting snp.\n"
            f"Algorithm: {model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: All features in the dataset except the target variable\n"
            f"Target variable: {target}\n"
            f"Trained on dataset: {args.data}\n"
            f"Model saved at; {args.models_dir}/trained/{model_name}.pkl\n"
            f"Performance metrics:\n"
            f"      - Accuracy: {accuracy:.2f}\n"
            f"      - Precision: {precision:.2f}\n"
            f"      - Recall : {recall:.2f}"
        )
        client.update_registered_model(name=model_name, description=description)

        # Add tags for better organization
        client.set_registered_model_tag(model_name, "algorithm", model_cfg["best_model"])
        client.set_registered_model_tag(model_name, "hyperparameters", str(model_cfg["parameters"]))
        client.set_registered_model_tag(model_name, "features", "All features except target variable")
        client.set_registered_model_tag(model_name, "target_variable", target)
        client.set_registered_model_tag(model_name, "training_dataset", args.data)
        client.set_registered_model_tag(model_name, "model_path", f"{args.models_dir}/trained/{model_name}.pkl")

        # Add dependency tags
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }

        for k,v in deps.items():
            client.set_registered_model_tag(model_name, k, v)

        # Save model locally
        save_path = f"{args.models_dir}/trained/{model_name}.pkl"
        joblib.dump(model, save_path)
        logger.info(f"Saved trained model to: {save_path}")
        logger.info(f"Final Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

        
        

