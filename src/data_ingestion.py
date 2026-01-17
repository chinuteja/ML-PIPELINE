import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# =========================
# PATH CONFIGURATION
# =========================
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING CONFIGURATION
# =========================
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(LOGS_DIR / "data_ingestion.log")
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


def load_data(data_url: str) -> pd.DataFrame:
    """Load dataset from URL."""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess dataset."""
    try:
        df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
        df = df.rename(columns={"v1": "target", "v2": "text"})
        logger.debug("Data preprocessing completed")
        return df
    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save train and test splits."""
    try:
        train_path = RAW_DATA_DIR / "train.csv"
        test_path = RAW_DATA_DIR / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.debug(
            "Train and test data saved to %s", RAW_DATA_DIR.resolve()
        )
    except Exception as e:
        logger.error("Failed to save data: %s", e)
        raise


# =========================
# MAIN PIPELINE STEP
# =========================
def main() -> None:
    try:
        DATA_URL = (
            "https://raw.githubusercontent.com/"
            "vikashishere/Datasets/main/spam.csv"
        )

        # TEST_SIZE = 0.2
        # RANDOM_STATE = 2
        params = load_params(params_path='params.yaml')
        TEST_SIZE = params['data_ingestion']['test_size']
        RANDOM_STATE = params['data_ingestion']['random_state']

        logger.debug("Project root: %s", PROJECT_ROOT.resolve())
        logger.debug("Raw data dir: %s", RAW_DATA_DIR.resolve())

        df = load_data(DATA_URL)
        df = preprocess_data(df)

        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        save_data(train_df, test_df)

    except Exception as e:
        logger.exception("Data ingestion failed")
        raise e


if __name__ == "__main__":
    main()
