"""FAISS vector storage with chunk provenance."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import faiss
import numpy as np
from numpy.typing import NDArray

from backend.exceptions import (
    EmptyVectorStoreError,
    VectorDimensionError,
    VectorStorePersistenceError,
)
from backend.models import DocumentChunk, RetrievedChunk


FloatMatrix = NDArray[np.float32]


class FaissVectorStore:
    """Store normalized vectors and their associated document chunks.

    FAISS stores only numeric vectors. Chunk metadata is maintained separately
    using the same insertion order as the vector index.
    """

    def __init__(self, *, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero."
            )

        self._dimension = dimension

        # Inner-product search is equivalent to cosine similarity when vectors
        # are normalized to unit length.
        self._index = faiss.IndexFlatIP(dimension)

        self._chunks: list[DocumentChunk] = []

    @property
    def dimension(self) -> int:
        """Return the configured vector dimension."""

        return self._dimension

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""

        return self._index.ntotal

    def add(
        self,
        *,
        chunks: list[DocumentChunk],
        embeddings: FloatMatrix,
    ) -> None:
        """Add chunks and their vectors in matching order."""

        if not chunks:
            raise ValueError(
                "At least one chunk is required."
            )

        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self._validate_matrix(
            matrix=matrix,
            expected_rows=len(chunks),
        )

        normalized_matrix = self._normalize_vectors(matrix)

        self._index.add(normalized_matrix)
        self._chunks.extend(chunks)

    def search(
        self,
        *,
        query_embedding: FloatMatrix,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most similar chunks ordered by score."""

        if self.size == 0:
            raise EmptyVectorStoreError(
                "Cannot search an empty vector store."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_matrix = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        self._validate_query_matrix(query_matrix)

        normalized_query = self._normalize_vectors(
            query_matrix
        )

        result_count = min(top_k, self.size)

        scores, indexes = self._index.search(
            normalized_query,
            result_count,
        )

        results: list[RetrievedChunk] = []

        for score, index_position in zip(
            scores[0],
            indexes[0],
        ):
            if index_position < 0:
                continue

            similarity_score = max(
                -1.0,
                min(1.0, float(score)),
            )
            if (
                minimum_score is not None
                and similarity_score < minimum_score
            ):
                continue

            results.append(
                RetrievedChunk(
                    chunk=self._chunks[index_position],
                    similarity_score=similarity_score,
                    rank=len(results) + 1,
                )
            )

        return results

    def remove_document(
        self,
        document_id: UUID,
    ) -> int:
        """Remove all chunks associated with one document.

        IndexFlatIP does not support convenient selective deletion with our
        simple positional metadata design, so the index is rebuilt.
        """

        retained_chunks = [
            chunk
            for chunk in self._chunks
            if chunk.document_id != document_id
        ]

        removed_count = len(self._chunks) - len(
            retained_chunks
        )

        if removed_count == 0:
            return 0

        retained_vectors = self._reconstruct_all_vectors()

        retained_positions = [
            index
            for index, chunk in enumerate(self._chunks)
            if chunk.document_id != document_id
        ]

        self._index = faiss.IndexFlatIP(
            self._dimension
        )
        self._chunks = retained_chunks

        if retained_positions:
            rebuilt_matrix = retained_vectors[
                retained_positions
            ]

            self._index.add(
                np.asarray(
                    rebuilt_matrix,
                    dtype=np.float32,
                )
            )

        return removed_count

    def save(
        self,
        *,
        index_path: Path,
        metadata_path: Path,
    ) -> None:
        """Persist the FAISS index and chunk metadata."""

        try:
            index_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            metadata_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            faiss.write_index(
                self._index,
                str(index_path),
            )

            metadata = {
                "dimension": self._dimension,
                "chunks": [
                    chunk.model_dump(
                        mode="json"
                    )
                    for chunk in self._chunks
                ],
            }

            metadata_path.write_text(
                json.dumps(
                    metadata,
                    indent=2,
                ),
                encoding="utf-8",
            )

        except (OSError, RuntimeError, TypeError) as exc:
            raise VectorStorePersistenceError(
                "The vector store could not be saved."
            ) from exc

    @classmethod
    def load(
        cls,
        *,
        index_path: Path,
        metadata_path: Path,
    ) -> "FaissVectorStore":
        """Load a persisted FAISS index and metadata."""

        try:
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Index file not found: {index_path}"
                )

            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata file not found: {metadata_path}"
                )

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            dimension = int(
                metadata["dimension"]
            )

            store = cls(
                dimension=dimension
            )

            loaded_index = faiss.read_index(
                str(index_path)
            )

            if loaded_index.d != dimension:
                raise VectorDimensionError(
                    "Persisted index dimension does not match metadata."
                )

            chunks = [
                DocumentChunk.model_validate(
                    chunk_data
                )
                for chunk_data in metadata["chunks"]
            ]

            if loaded_index.ntotal != len(chunks):
                raise VectorStorePersistenceError(
                    "Vector count does not match chunk metadata count."
                )

            store._index = loaded_index
            store._chunks = chunks

            return store

        except VectorDimensionError:
            raise

        except VectorStorePersistenceError:
            raise

        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            raise VectorStorePersistenceError(
                "The vector store could not be loaded."
            ) from exc

    def _validate_matrix(
        self,
        *,
        matrix: FloatMatrix,
        expected_rows: int,
    ) -> None:
        """Validate vectors before indexing."""

        if matrix.ndim != 2:
            raise VectorDimensionError(
                "Embeddings must be a two-dimensional matrix."
            )

        if matrix.shape[0] != expected_rows:
            raise VectorDimensionError(
                "Embedding count must match chunk count."
            )

        if matrix.shape[1] != self._dimension:
            raise VectorDimensionError(
                "Embedding dimension does not match vector store dimension."
            )

        if not np.isfinite(matrix).all():
            raise VectorDimensionError(
                "Embeddings contain non-finite values."
            )

    def _validate_query_matrix(
        self,
        matrix: FloatMatrix,
    ) -> None:
        """Require one query vector with the correct dimension."""

        if matrix.ndim != 2:
            raise VectorDimensionError(
                "Query embedding must be two-dimensional."
            )

        if matrix.shape != (1, self._dimension):
            raise VectorDimensionError(
                "Query embedding must contain exactly one vector "
                "with the configured dimension."
            )

        if not np.isfinite(matrix).all():
            raise VectorDimensionError(
                "Query embedding contains non-finite values."
            )

    @staticmethod
    def _normalize_vectors(
        matrix: FloatMatrix,
    ) -> FloatMatrix:
        """Return unit-normalized float32 vectors."""

        normalized = np.array(
            matrix,
            dtype=np.float32,
            copy=True,
        )

        norms = np.linalg.norm(
            normalized,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise VectorDimensionError(
                "Zero-length vectors cannot be indexed."
            )

        normalized /= norms

        return normalized

    def _reconstruct_all_vectors(
        self,
    ) -> FloatMatrix:
        """Reconstruct all vectors from the flat FAISS index."""

        if self.size == 0:
            return np.empty(
                (0, self._dimension),
                dtype=np.float32,
            )

        vectors = [
            self._index.reconstruct(index)
            for index in range(self.size)
        ]

        return np.asarray(
            vectors,
            dtype=np.float32,
        )