"""Deterministic embedding implementation for unit tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from backend.exceptions import EmptyEmbeddingInputError
from backend.services.embedding_service import EmbeddingMatrix


class DeterministicFakeEmbeddingService:
    """Create repeatable vectors without loading an ML model.

    This implementation is suitable for unit tests, not semantic retrieval in
    production.
    """

    def __init__(self, dimension: int = 8) -> None:
        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero."
            )

        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the configured fake-vector dimension."""

        return self._dimension

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> EmbeddingMatrix:
        """Generate deterministic normalized vectors."""

        if not texts:
            raise EmptyEmbeddingInputError(
                "At least one text is required."
            )

        vectors: list[np.ndarray] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise EmptyEmbeddingInputError(
                    f"Text at index {index} cannot be empty."
                )

            vectors.append(
                self._vector_from_text(text.strip())
            )

        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, query: str) -> EmbeddingMatrix:
        """Generate one deterministic query vector."""

        if not query or not query.strip():
            raise EmptyEmbeddingInputError(
                "The query cannot be empty."
            )

        return self.embed_texts([query])

    def _vector_from_text(self, text: str) -> np.ndarray:
        """Convert a stable hash into a normalized vector."""

        values: list[float] = []
        counter = 0

        while len(values) < self._dimension:
            digest = hashlib.sha256(
                f"{text}:{counter}".encode("utf-8")
            ).digest()

            for byte in digest:
                value = (float(byte) / 127.5) - 1.0
                values.append(value)

                if len(values) == self._dimension:
                    break

            counter += 1

        vector = np.asarray(
            values,
            dtype=np.float32,
        )

        norm = np.linalg.norm(vector)

        if norm == 0:
            vector[0] = 1.0
            norm = 1.0

        return vector / norm