import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


# =========================
# PATH CONFIGURATION
# =========================
PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING CONFIGURATION
# =========================
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(LOGS_DIR / "model_building.log")
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


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load processed training data."""
    try:
        df = pd.read_csv(csv_path)
        logger.debug(
            "Data loaded from %s with shape %s",
            csv_path.resolve(),
            df.shape,
        )
        return df
    except Exception as e:
        logger.error("Failed to load data from %s: %s", csv_path, e)
        raise


def train_model(X_train: np.ndarray,y_train: np.ndarray,params: dict,) -> RandomForestClassifier:
    """Train RandomForest model."""
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must have the same number of samples"
            )

        logger.debug("Training RandomForest with params: %s", params)

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            random_state=params["random_state"],
        )

        model.fit(X_train, y_train)
        logger.debug("Model training completed")

        return model

    except Exception as e:
        logger.error("Model training failed: %s", e)
        raise


def save_model(model: RandomForestClassifier, path: Path) -> None:
    """Persist trained model."""
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.debug("Model saved to %s", path.resolve())
    except Exception as e:
        logger.error("Failed to save model: %s", e)
        raise


# =========================
# MAIN PIPELINE STEP
# =========================
def main() -> None:
    try:
        # Params (hardcoded for now; wire params.yaml later)
        # MODEL_PARAMS = {
        #     "n_estimators": 100,
        #     "random_state": 42,
        # }
        MODEL_PARAMS = load_params('params.yaml')['model_building']
        logger.debug("Loading TF-IDF training data")

        train_df = load_data(
            PROCESSED_DATA_DIR / "train_tfidf.csv"
        )

        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values

        model = train_model(X_train, y_train, MODEL_PARAMS)

        model_path = MODELS_DIR / "model.pkl"
        save_model(model, model_path)

    except Exception:
        logger.exception("Model building failed")
        raise


if __name__ == "__main__":
    main()
