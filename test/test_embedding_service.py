"""Tests for embedding service contracts."""

import numpy as np
import pytest

from backend.exceptions import EmptyEmbeddingInputError
from backend.services.embedding_service import TextEmbedder
from backend.services.fake_embedding_service import (
    DeterministicFakeEmbeddingService,
)


def test_fake_embedder_satisfies_protocol():
    """The fake implementation must satisfy the application contract."""

    service = DeterministicFakeEmbeddingService(dimension=8)

    assert isinstance(service, TextEmbedder)


def test_embed_texts_preserves_input_count():
    """Each text should produce exactly one vector."""

    service = DeterministicFakeEmbeddingService(dimension=8)

    vectors = service.embed_texts(
        [
            "The firm must retain customer records.",
            "Records are reviewed annually.",
        ]
    )

    assert vectors.shape == (2, 8)
    assert vectors.dtype == np.float32


def test_embeddings_are_deterministic():
    """Identical text should produce identical vectors."""

    service = DeterministicFakeEmbeddingService(dimension=12)

    first = service.embed_texts(["Customer records must be retained."])

    second = service.embed_texts(["Customer records must be retained."])

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_embeddings_are_normalized():
    """Vectors should have approximately unit L2 length."""

    service = DeterministicFakeEmbeddingService(dimension=16)

    vectors = service.embed_texts(
        [
            "First policy passage.",
            "Second policy passage.",
        ]
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    np.testing.assert_allclose(
        norms,
        np.ones(2),
        rtol=1e-5,
        atol=1e-6,
    )


def test_different_text_produces_different_vectors():
    """Different text should not generate identical fake vectors."""

    service = DeterministicFakeEmbeddingService(dimension=8)

    vectors = service.embed_texts(
        [
            "Customer records.",
            "Annual compliance review.",
        ]
    )

    assert not np.array_equal(
        vectors[0],
        vectors[1],
    )


def test_embed_query_returns_one_row():
    """Query embedding should use the same matrix contract."""

    service = DeterministicFakeEmbeddingService(dimension=10)

    vector = service.embed_query("What is the record retention requirement?")

    assert vector.shape == (1, 10)


@pytest.mark.parametrize(
    "texts",
    [
        [],
        [""],
        ["   "],
        ["valid text", ""],
    ],
)
def test_empty_embedding_inputs_are_rejected(texts):
    """Empty text must never be indexed."""

    service = DeterministicFakeEmbeddingService()

    with pytest.raises(
        EmptyEmbeddingInputError,
    ):
        service.embed_texts(texts)


def test_invalid_dimension_is_rejected():
    """Embedding vectors require at least one dimension."""

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        DeterministicFakeEmbeddingService(dimension=0)
