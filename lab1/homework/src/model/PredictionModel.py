from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
from typing import Optional
from pathlib import Path


def get_current_dir_path() -> Path:
    """
    Get the path of the current file.
    """
    return Path(__file__).parent.resolve()


def load_sentence_transformer() -> SentenceTransformer:
    """
    Load the trained SentenceTransformer model from the saved_models folder.
    """
    current_dir = get_current_dir_path()
    path = Path(f"{current_dir}/saved_models/sentence_transformer.model")

    if not path.exists():
        raise FileNotFoundError(f"Sentence Transformer model not found at: {path}")
    return SentenceTransformer(str(path))


def load_classifier() -> LogisticRegression:
    """
    Load the trained LogisticRegression classifier from the saved_models folder.
    """
    current_dir = get_current_dir_path()
    path = Path(f"{current_dir}/saved_models/classifier.joblib")

    if not path.exists():
        raise FileNotFoundError(f"Classifier not found at: {path}")

    return joblib.load(str(path))


def map_to_class(label: int) -> Optional[str]:
    """
    Map a numeric sentiment label to a string.
    """
    return {0: "negative", 1: "neutral", 2: "positive"}.get(label)


class PredictionModel:
    """
    Wrapper class that combines a SentenceTransformer and a classifier for sentiment prediction.
    """

    def __init__(self):
        self.sentence_transformer = load_sentence_transformer()
        self.classifier = load_classifier()

    def predict(self, sentence: str) -> str:
        embedding = self.sentence_transformer.encode(sentence)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        label = self.classifier.predict(embedding)[0]
        sentiment = map_to_class(label)

        if sentiment is None:
            raise ValueError(f"Invalid sentiment label: {label}")
        return sentiment
