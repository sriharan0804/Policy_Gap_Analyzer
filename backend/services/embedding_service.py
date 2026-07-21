"""Text embedding abstractions and production implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from backend.exceptions import (
    EmbeddingModelError,
    EmptyEmbeddingInputError,
)

EmbeddingMatrix = NDArray[np.float32]


@runtime_checkable
class TextEmbedder(Protocol):
    """Contract implemented by any text embedding provider."""

    @property
    def dimension(self) -> int:
        """Return the number of values in each embedding vector."""

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> EmbeddingMatrix:
        """Generate one normalized vector for each supplied text."""

    def embed_query(self, query: str) -> EmbeddingMatrix:
        """Generate one normalized query vector."""


class SentenceTransformerEmbeddingService:
    """Generate normalized embeddings using Sentence Transformers.

    The model is imported and loaded lazily so that modules and unit tests can
    run without downloading or initializing a transformer model.
    """

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model = None
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Return the model embedding dimension."""

        if self._dimension is None:
            model = self._get_model()
            dimension = model.get_sentence_embedding_dimension()

            if dimension is None or dimension <= 0:
                raise EmbeddingModelError(
                    "The embedding model did not report a valid dimension."
                )

            self._dimension = int(dimension)

        return self._dimension

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> EmbeddingMatrix:
        """Generate normalized float32 vectors for multiple texts."""

        validated_texts = self._validate_texts(texts)
        model = self._get_model()

        try:
            vectors = model.encode(
                validated_texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingModelError(
                "The embedding model failed to encode the supplied text."
            ) from exc

        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        self._validate_embedding_matrix(
            matrix=matrix,
            expected_rows=len(validated_texts),
        )

        if self._dimension is None:
            self._dimension = int(matrix.shape[1])

        return matrix

    def embed_query(self, query: str) -> EmbeddingMatrix:
        """Generate a normalized matrix containing one query vector."""

        if not query or not query.strip():
            raise EmptyEmbeddingInputError("The query cannot be empty.")

        return self.embed_texts([query])

    def _get_model(self):
        """Load the transformer model only when first needed."""

        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )

        except Exception as exc:
            raise EmbeddingModelError(
                f"Could not load embedding model: {self._model_name}."
            ) from exc

        return self._model

    @staticmethod
    def _validate_texts(
        texts: Sequence[str],
    ) -> list[str]:
        """Return validated texts while preserving order."""

        if not texts:
            raise EmptyEmbeddingInputError("At least one text is required.")

        validated: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmptyEmbeddingInputError(
                    f"Text at index {index} must be a string."
                )

            normalized_text = text.strip()

            if not normalized_text:
                raise EmptyEmbeddingInputError(
                    f"Text at index {index} cannot be empty."
                )

            validated.append(normalized_text)

        return validated

    @staticmethod
    def _validate_embedding_matrix(
        *,
        matrix: EmbeddingMatrix,
        expected_rows: int,
    ) -> None:
        """Validate shape and numeric integrity of generated vectors."""

        if matrix.ndim != 2:
            raise EmbeddingModelError(
                "The embedding result must be a two-dimensional matrix."
            )

        if matrix.shape[0] != expected_rows:
            raise EmbeddingModelError(
                "The number of embeddings does not match the input count."
            )

        if matrix.shape[1] <= 0:
            raise EmbeddingModelError("Embedding vectors cannot have zero dimensions.")

        if not np.isfinite(matrix).all():
            raise EmbeddingModelError("Embedding vectors contain non-finite values.")
