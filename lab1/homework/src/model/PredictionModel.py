from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
from typing import Optional
from pathlib import Path


def get_current_dir_path() -> Path:
    return Path(__file__).parent.resolve()


def load_sentence_transformer() -> SentenceTransformer:
    current_dir = get_current_dir_path()
    path = Path(f"{current_dir}/saved_models/sentence_transformer.model")

    if not path.exists():
        raise FileNotFoundError(f"Sentence Transformer model not found at: {path}")
    return SentenceTransformer(str(path))


def load_classifier() -> LogisticRegression:
    current_dir = get_current_dir_path()
    path = Path(f"{current_dir}/saved_models/classifier.joblib")

    if not path.exists():
        raise FileNotFoundError(f"Classifier not found at: {path}")

    return joblib.load(str(path))


def map_to_class(sentiment: int) -> Optional[str]:
    return {0: "negative", 1: "neutral", 2: "positive"}.get(sentiment)


class PredictionModel:
    def __init__(self):
        self.sentence_transformer = load_sentence_transformer()
        self.classifier = load_classifier()

    def predict(self, sentence: str) -> str:
        embedding = self.sentence_transformer.encode(sentence)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        sentiment = self.classifier.predict(embedding)[0]
        label = map_to_class(sentiment)

        if not label:
            raise ValueError(f"Invalid sentiment label: {sentiment}")
        return label
