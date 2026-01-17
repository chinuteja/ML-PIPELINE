import json
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yaml
from dvclive import Live
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================
# PATH CONFIGURATION
# =========================
PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
DVC_LIVE_DIR = PROJECT_ROOT / "dvclive"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DVC_LIVE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING CONFIGURATION
# =========================
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(LOGS_DIR / "model_evaluation.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# =========================
# UTILS
# =========================
def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from YAML."""
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Failed to load params: %s", e)
        raise


def load_model(model_path: Path):
    """Load trained model."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.debug("Model loaded from %s", model_path.resolve())
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load test dataset."""
    try:
        df = pd.read_csv(csv_path)
        logger.debug("Data loaded from %s", csv_path.resolve())
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }

        logger.debug("Evaluation metrics calculated: %s", metrics)
        return metrics

    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        raise


def save_metrics(metrics: dict, path: Path) -> None:
    """Persist metrics as JSON."""
    try:
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Metrics saved to %s", path.resolve())
    except Exception as e:
        logger.error("Failed to save metrics: %s", e)
        raise


# =========================
# MAIN PIPELINE STEP
# =========================
def main() -> None:
    try:
        # Params (hardcoded for now; wire params.yaml later)
        MODEL_PARAMS = {
            "n_estimators": 100,
            "random_state": 42,
        }

        logger.debug("Loading model and test data")

        model = load_model(MODELS_DIR / "model.pkl")
        test_df = load_data(PROCESSED_DATA_DIR / "test_tfidf.csv")

        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        metrics = evaluate_model(model, X_test, y_test)

        # =========================
        # DVC LIVE TRACKING
        # =========================
        with Live(dir=DVC_LIVE_DIR, save_dvc_exp=True) as live:
            for name, value in metrics.items():
                live.log_metric(name, value)

            live.log_params(MODEL_PARAMS)

        save_metrics(metrics, REPORTS_DIR / "metrics.json")

    except Exception:
        logger.exception("Model evaluation failed")
        raise


if __name__ == "__main__":
    main()
