import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# PATH CONFIGURATION
# =========================
PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING CONFIGURATION
# =========================
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(LOGS_DIR / "feature_engineering.log")
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
    """Load CSV and fill missing values."""
    try:
        df = pd.read_csv(csv_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded from %s", csv_path.resolve())
        return df
    except Exception as e:
        logger.error("Failed to load data from %s: %s", csv_path, e)
        raise


def apply_tfidf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply TF-IDF vectorization."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_df["text"].values
        y_train = train_df["target"].values

        X_test = test_df["text"].values
        y_test = test_df["target"].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_out = pd.DataFrame(X_train_tfidf.toarray())
        train_out["label"] = y_train

        test_out = pd.DataFrame(X_test_tfidf.toarray())
        test_out["label"] = y_test

        logger.debug("TF-IDF transformation completed")
        return train_out, test_out

    except Exception as e:
        logger.error("TF-IDF transformation failed: %s", e)
        raise


# =========================
# MAIN PIPELINE STEP
# =========================
def main() -> None:
    try:
        # Params (hardcoded for now, wire params.yaml later)
        MAX_FEATURES = 50

        logger.debug("Loading interim datasets")

        train_df = load_data(INTERIM_DATA_DIR / "train_processed.csv")
        test_df = load_data(INTERIM_DATA_DIR / "test_processed.csv")

        train_tfidf, test_tfidf = apply_tfidf(
            train_df, test_df, MAX_FEATURES
        )

        train_tfidf.to_csv(
            PROCESSED_DATA_DIR / "train_tfidf.csv",
            index=False,
        )
        test_tfidf.to_csv(
            PROCESSED_DATA_DIR / "test_tfidf.csv",
            index=False,
        )

        logger.debug(
            "TF-IDF features saved to %s",
            PROCESSED_DATA_DIR.resolve(),
        )

    except Exception:
        logger.exception("Feature engineering failed")
        raise


if __name__ == "__main__":
    main()
