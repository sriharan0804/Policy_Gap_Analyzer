"""Tests for the retrieval service."""

from uuid import uuid4

import pytest

from backend.exceptions import (
    EmptyEmbeddingInputError,
    EmptyVectorStoreError,
)
from backend.models import DocumentChunk
from backend.services.fake_embedding_service import (
    DeterministicFakeEmbeddingService,
)
from backend.services.retrieval_service import (
    RetrievalService,
)
from backend.services.vector_store import (
    FaissVectorStore,
)


def make_chunk(
    *,
    text: str,
    chunk_index: int,
    page_number: int = 1,
) -> DocumentChunk:
    """Create a valid document chunk for retrieval tests."""

    return DocumentChunk(
        document_id=uuid4(),
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        character_count=len(text),
        start_character=0,
        end_character=len(text),
    )


def build_retrieval_service(
    chunks: list[DocumentChunk],
    *,
    dimension: int = 16,
) -> RetrievalService:
    """Build a retrieval service with indexed fake embeddings."""

    embedder = DeterministicFakeEmbeddingService(dimension=dimension)

    vector_store = FaissVectorStore(dimension=dimension)

    if chunks:
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

        vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
        )

    return RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
    )


def test_retrieve_returns_matching_chunk_first():
    """An identical query should return the matching chunk first."""

    chunks = [
        make_chunk(
            text="Customer records must be retained for five years.",
            chunk_index=0,
        ),
        make_chunk(
            text="The annual compliance meeting occurs in December.",
            chunk_index=1,
        ),
    ]

    service = build_retrieval_service(chunks)

    results = service.retrieve(
        "Customer records must be retained for five years.",
        top_k=2,
        minimum_score=None,
    )

    assert len(results) == 2
    assert results[0].chunk.text == (
        "Customer records must be retained for five years."
    )
    assert results[0].rank == 1


def test_retrieve_respects_top_k():
    """The result count should not exceed top_k."""

    chunks = [
        make_chunk(
            text=f"Policy passage number {index}.",
            chunk_index=index,
        )
        for index in range(5)
    ]

    service = build_retrieval_service(chunks)

    results = service.retrieve(
        "Policy passage number 2.",
        top_k=2,
        minimum_score=None,
    )

    assert len(results) == 2


def test_results_are_sorted_by_similarity():
    """Results should be ordered from highest to lowest score."""

    chunks = [
        make_chunk(
            text="First regulatory passage.",
            chunk_index=0,
        ),
        make_chunk(
            text="Second internal policy passage.",
            chunk_index=1,
        ),
        make_chunk(
            text="Third supervisory procedure.",
            chunk_index=2,
        ),
    ]

    service = build_retrieval_service(chunks)

    results = service.retrieve(
        "Second internal policy passage.",
        top_k=3,
        minimum_score=None,
    )

    scores = [result.similarity_score for result in results]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    assert [result.rank for result in results] == [1, 2, 3]


def test_minimum_score_filters_results():
    """Results below the configured threshold should be removed."""

    chunks = [
        make_chunk(
            text="The firm must retain customer records.",
            chunk_index=0,
        ),
        make_chunk(
            text="The cafeteria closes at six.",
            chunk_index=1,
        ),
    ]

    service = build_retrieval_service(chunks)

    results = service.retrieve(
        "The firm must retain customer records.",
        top_k=2,
        minimum_score=0.99,
    )

    assert len(results) == 1
    assert results[0].similarity_score >= 0.99


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_empty_query_is_rejected(query):
    """Blank retrieval queries must fail before vector search."""

    service = build_retrieval_service(
        [
            make_chunk(
                text="Valid policy text.",
                chunk_index=0,
            )
        ]
    )

    with pytest.raises(
        EmptyEmbeddingInputError,
    ):
        service.retrieve(query)


def test_non_string_query_is_rejected():
    """A query must be supplied as text."""

    service = build_retrieval_service(
        [
            make_chunk(
                text="Valid policy text.",
                chunk_index=0,
            )
        ]
    )

    with pytest.raises(
        EmptyEmbeddingInputError,
    ):
        service.retrieve(123)  # type: ignore[arg-type]


def test_empty_vector_store_raises_error():
    """Retrieval from an empty index should fail clearly."""

    service = build_retrieval_service([])

    with pytest.raises(
        EmptyVectorStoreError,
    ):
        service.retrieve("record retention requirement")


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        -10,
    ],
)
def test_invalid_top_k_is_rejected(top_k):
    """top_k must always be positive."""

    service = build_retrieval_service(
        [
            make_chunk(
                text="Valid policy text.",
                chunk_index=0,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        service.retrieve(
            "policy",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "minimum_score",
    [
        -1.1,
        1.1,
        5.0,
    ],
)
def test_invalid_minimum_score_is_rejected(
    minimum_score,
):
    """Cosine-style thresholds must remain within valid bounds."""

    service = build_retrieval_service(
        [
            make_chunk(
                text="Valid policy text.",
                chunk_index=0,
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="minimum_score",
    ):
        service.retrieve(
            "policy",
            minimum_score=minimum_score,
        )


def test_dimension_mismatch_is_rejected():
    """Embedder and vector store dimensions must match."""

    embedder = DeterministicFakeEmbeddingService(dimension=8)

    vector_store = FaissVectorStore(dimension=16)

    with pytest.raises(
        ValueError,
        match="dimension",
    ):
        RetrievalService(
            embedder=embedder,
            vector_store=vector_store,
        )
