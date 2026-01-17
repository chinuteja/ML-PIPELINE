import logging
from pathlib import Path
import string

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder


# =========================
# NLTK SETUP (idempotent)
# =========================
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


# =========================
# PATH CONFIGURATION
# =========================
PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
LOGS_DIR = PROJECT_ROOT / "logs"

INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING CONFIGURATION
# =========================
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(LOGS_DIR / "data_preprocessing.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# =========================
# TEXT TRANSFORMATION
# =========================
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def transform_text(text: str) -> str:
    """
    Normalize text:
    - lowercase
    - tokenize
    - remove stopwords, punctuation, non-alphanumerics
    - stem
    """
    tokens = nltk.word_tokenize(text.lower())
    tokens = [
        ps.stem(word)
        for word in tokens
        if word.isalnum()
        and word not in stop_words
        and word not in string.punctuation
    ]
    return " ".join(tokens)


# =========================
# DATAFRAME PREPROCESSING
# =========================
def preprocess_df(
    df: pd.DataFrame,
    text_column: str = "text",
    target_column: str = "target",
) -> pd.DataFrame:
    try:
        logger.debug("Starting preprocessing")

        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded")

        df = df.drop_duplicates(keep="first")
        logger.debug("Duplicates removed")

        df[text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")

        return df

    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        raise


# =========================
# MAIN PIPELINE STEP
# =========================
def main() -> None:
    try:
        logger.debug("Loading raw datasets")

        train_df = pd.read_csv(RAW_DATA_DIR / "train.csv")
        test_df = pd.read_csv(RAW_DATA_DIR / "test.csv")

        train_processed = preprocess_df(train_df)
        test_processed = preprocess_df(test_df)

        train_processed.to_csv(
            INTERIM_DATA_DIR / "train_processed.csv",
            index=False,
        )
        test_processed.to_csv(
            INTERIM_DATA_DIR / "test_processed.csv",
            index=False,
        )

        logger.debug(
            "Processed data saved to %s",
            INTERIM_DATA_DIR.resolve(),
        )

    except Exception:
        logger.exception("Data preprocessing failed")
        raise


if __name__ == "__main__":
    main()
