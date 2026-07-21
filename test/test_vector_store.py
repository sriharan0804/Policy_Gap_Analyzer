"""Tests for the FAISS vector store."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from backend.exceptions import (
    EmptyVectorStoreError,
    VectorDimensionError,
)
from backend.models import DocumentChunk
from backend.services.fake_embedding_service import (
    DeterministicFakeEmbeddingService,
)
from backend.services.vector_store import (
    FaissVectorStore,
)


def make_chunk(
    *,
    document_id,
    chunk_index: int,
    text: str,
    page_number: int = 1,
) -> DocumentChunk:
    """Create a valid chunk for tests."""

    return DocumentChunk(
        document_id=document_id,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        character_count=len(text),
        start_character=0,
        end_character=len(text),
    )


def test_add_vectors_and_search():
    """Indexed chunks should be returned with provenance."""

    document_id = uuid4()

    chunks = [
        make_chunk(
            document_id=document_id,
            chunk_index=0,
            text="Customer records must be retained.",
        ),
        make_chunk(
            document_id=document_id,
            chunk_index=1,
            text="The policy is reviewed annually.",
        ),
    ]

    embedder = DeterministicFakeEmbeddingService(dimension=8)

    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    store = FaissVectorStore(dimension=8)

    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    query = embedder.embed_query("Customer records must be retained.")

    results = store.search(
        query_embedding=query,
        top_k=2,
    )

    assert store.size == 2
    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].chunk.text == ("Customer records must be retained.")


def test_search_preserves_page_and_document_metadata():
    """Retrieval results must retain citation provenance."""

    document_id = uuid4()

    chunk = make_chunk(
        document_id=document_id,
        page_number=7,
        chunk_index=4,
        text="Supervisory review is required.",
    )

    embedder = DeterministicFakeEmbeddingService(dimension=8)

    store = FaissVectorStore(dimension=8)

    store.add(
        chunks=[chunk],
        embeddings=embedder.embed_texts([chunk.text]),
    )

    results = store.search(query_embedding=embedder.embed_query(chunk.text))

    assert results[0].chunk.document_id == document_id
    assert results[0].chunk.page_number == 7
    assert results[0].chunk.chunk_index == 4


def test_empty_store_cannot_be_searched():
    """Searching an empty index should fail clearly."""

    store = FaissVectorStore(dimension=8)

    query = np.ones(
        (1, 8),
        dtype=np.float32,
    )

    with pytest.raises(
        EmptyVectorStoreError,
    ):
        store.search(query_embedding=query)


def test_wrong_embedding_dimension_is_rejected():
    """Indexed vectors must match the configured dimension."""

    chunk = make_chunk(
        document_id=uuid4(),
        chunk_index=0,
        text="Policy text.",
    )

    store = FaissVectorStore(dimension=8)

    wrong_vectors = np.ones(
        (1, 6),
        dtype=np.float32,
    )

    with pytest.raises(
        VectorDimensionError,
        match="dimension",
    ):
        store.add(
            chunks=[chunk],
            embeddings=wrong_vectors,
        )


def test_chunk_count_must_match_vector_count():
    """Every vector must have exactly one metadata record."""

    chunks = [
        make_chunk(
            document_id=uuid4(),
            chunk_index=0,
            text="First chunk.",
        ),
        make_chunk(
            document_id=uuid4(),
            chunk_index=1,
            text="Second chunk.",
        ),
    ]

    store = FaissVectorStore(dimension=8)

    one_vector = np.ones(
        (1, 8),
        dtype=np.float32,
    )

    with pytest.raises(
        VectorDimensionError,
        match="count",
    ):
        store.add(
            chunks=chunks,
            embeddings=one_vector,
        )


def test_save_and_load_store(tmp_path: Path):
    """Persisted vectors must remain connected to metadata."""

    document_id = uuid4()

    chunk = make_chunk(
        document_id=document_id,
        page_number=5,
        chunk_index=0,
        text="Records must be retained for five years.",
    )

    embedder = DeterministicFakeEmbeddingService(dimension=8)

    store = FaissVectorStore(dimension=8)

    store.add(
        chunks=[chunk],
        embeddings=embedder.embed_texts([chunk.text]),
    )

    index_path = tmp_path / "chunks.faiss"
    metadata_path = tmp_path / "chunks.json"

    store.save(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    loaded = FaissVectorStore.load(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    results = loaded.search(query_embedding=embedder.embed_query(chunk.text))

    assert loaded.size == 1
    assert results[0].chunk.document_id == document_id
    assert results[0].chunk.page_number == 5


def test_remove_document_rebuilds_index():
    """Removing one document must retain other indexed documents."""

    first_document = uuid4()
    second_document = uuid4()

    chunks = [
        make_chunk(
            document_id=first_document,
            chunk_index=0,
            text="First document requirement.",
        ),
        make_chunk(
            document_id=second_document,
            chunk_index=1,
            text="Second document policy.",
        ),
    ]

    embedder = DeterministicFakeEmbeddingService(dimension=8)

    store = FaissVectorStore(dimension=8)

    store.add(
        chunks=chunks,
        embeddings=embedder.embed_texts([chunk.text for chunk in chunks]),
    )

    removed = store.remove_document(first_document)

    assert removed == 1
    assert store.size == 1

    results = store.search(
        query_embedding=embedder.embed_query("Second document policy.")
    )

    assert results[0].chunk.document_id == second_document


def test_similarity_score_is_clamped_to_valid_range():
    """Floating-point rounding must not produce scores above one."""

    chunk = make_chunk(
        document_id=uuid4(),
        chunk_index=0,
        text="Identical text for similarity testing.",
    )

    embedder = DeterministicFakeEmbeddingService(dimension=16)

    store = FaissVectorStore(dimension=16)

    embedding = embedder.embed_texts([chunk.text])

    store.add(
        chunks=[chunk],
        embeddings=embedding,
    )

    results = store.search(
        query_embedding=embedding,
        top_k=1,
    )

    assert results[0].similarity_score <= 1.0
    assert results[0].similarity_score >= -1.0
