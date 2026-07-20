"""Semantic retrieval orchestration service."""

from __future__ import annotations

from backend.exceptions import EmptyEmbeddingInputError
from backend.models import RetrievedChunk
from backend.services.embedding_service import TextEmbedder
from backend.services.vector_store import FaissVectorStore


class RetrievalService:
    """Coordinate query embedding and vector similarity search."""

    def __init__(
        self,
        *,
        embedder: TextEmbedder,
        vector_store: FaissVectorStore,
    ) -> None:
        if embedder.dimension != vector_store.dimension:
            raise ValueError(
                "Embedder dimension must match vector store dimension."
            )

        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        minimum_score: float | None = 0.35,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a natural-language query."""

        validated_query = self._validate_query(query)

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if minimum_score is not None:
            if minimum_score < -1.0 or minimum_score > 1.0:
                raise ValueError(
                    "minimum_score must be between -1.0 and 1.0."
                )

        query_embedding = self._embedder.embed_query(
            validated_query
        )

        return self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            minimum_score=minimum_score,
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate and normalize a retrieval query."""

        if not isinstance(query, str):
            raise EmptyEmbeddingInputError(
                "The retrieval query must be a string."
            )

        normalized_query = query.strip()

        if not normalized_query:
            raise EmptyEmbeddingInputError(
                "The retrieval query cannot be empty."
            )

        return normalized_query