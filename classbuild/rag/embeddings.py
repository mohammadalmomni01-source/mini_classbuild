import logging
from functools import lru_cache
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEVICE = "cpu"
BATCH_SIZE = 32


class EmbeddingError(Exception):
    """Base class. Every message is written for the user."""


class EmbeddingInputError(EmbeddingError):
    pass


class EmbeddingModelUnavailableError(EmbeddingError):
    pass


class EmbeddingResourceError(EmbeddingError):
    pass


class InvalidEmbeddingError(EmbeddingError):
    pass


class TextEncoder(Protocol):
    """The one method we need. The real model and test fakes both have it."""

    def encode(self, sentences, **kwargs): ...


@lru_cache(maxsize=2)
def load_model(model_name=MODEL_NAME, device=DEVICE):
    """Loads the model once per process. The first run downloads about 90 MB."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingModelUnavailableError(
            "The embedding library is not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc

    try:
        model = SentenceTransformer(model_name, device=device)
    except MemoryError as exc:
        raise EmbeddingResourceError("Not enough memory to load the embedding model.") from exc
    except Exception as exc:
        logger.exception("Could not load the embedding model")
        raise EmbeddingModelUnavailableError(
            "The embedding model could not be loaded. The first run needs an "
            "internet connection to download it (about 90 MB)."
        ) from exc

    dimension = model.get_embedding_dimension()
    if dimension != EMBEDDING_DIM: 
        raise EmbeddingModelUnavailableError(
            f"The model produces {dimension}-number vectors but {EMBEDDING_DIM} were expected."
        )
    return model


def validate_embedding_matrix(matrix, expected_rows=None, dimension=EMBEDDING_DIM):
    try:
        array = np.asarray(matrix, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidEmbeddingError("The embedding model returned unusable data.") from exc

    if array.ndim != 2 or array.shape[1] != dimension:
        raise InvalidEmbeddingError("The embedding model returned vectors of the wrong size.")
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise InvalidEmbeddingError("The embedding model returned the wrong number of vectors.")
    if not np.isfinite(array).all():
        raise InvalidEmbeddingError("The embedding model returned invalid numbers.")
    if (np.linalg.norm(array, axis=1) == 0).any():
        raise InvalidEmbeddingError("The embedding model returned an empty vector.")
    return array


class EmbeddingService:
    def __init__(self, model=None, model_name=MODEL_NAME):
        self._model = model
        self.model_name = model_name

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.model_name)
        return self._model

    def embed_texts(self, texts):
        texts = list(texts)
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingInputError("There is no text to embed.")
        try:
            vectors = self.model.encode(
                texts,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except EmbeddingError:
            raise
        except MemoryError as exc:
            raise EmbeddingResourceError("Not enough memory to embed this text.") from exc
        except Exception as exc:
            logger.exception("Embedding failed")
            raise EmbeddingModelUnavailableError(
                "The text could not be embedded. Please try again."
            ) from exc
        return validate_embedding_matrix(vectors, expected_rows=len(texts))

    def embed_query(self, query):
        if not isinstance(query, str) or not query.strip():
            raise EmbeddingInputError("Please enter a question first.")
        return self.embed_texts([query.strip()])[0]