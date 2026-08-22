from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embeddings.embedding_service import get_embedding_model


# Persistent location for the application's vector database.
CHROMA_PERSIST_DIRECTORY = Path("data/chroma")


# Collection name for the RAG application.
CHROMA_COLLECTION_NAME = "genai_capstone_documents"


def get_vector_store() -> Chroma:
    """
    Create or connect to the persistent ChromaDB vector store.
    """

    CHROMA_PERSIST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_model = get_embedding_model()

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_PERSIST_DIRECTORY),
    )


def add_documents(
    documents: list[Document],
) -> list[str]:
    """
    Add documents to the ChromaDB vector store.

    Returns the IDs generated for the stored documents.
    """

    if not documents:
        return []

    vector_store = get_vector_store()

    document_ids = vector_store.add_documents(
        documents
    )

    return document_ids


def similarity_search(
    query: str,
    k: int = 4,
) -> list[Document]:
    """
    Retrieve the most relevant documents for a query.
    """

    if not query or not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    if k <= 0:
        raise ValueError(
            "Number of results k must be greater than zero."
        )

    vector_store = get_vector_store()

    return vector_store.similarity_search(
        query,
        k=k,
    )


def similarity_search_with_scores(
    query: str,
    k: int = 4,
) -> list[tuple[Document, float]]:
    """
    Retrieve relevant documents together with
    their similarity scores.
    """

    if not query or not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    if k <= 0:
        raise ValueError(
            "Number of results k must be greater than zero."
        )

    vector_store = get_vector_store()

    return vector_store.similarity_search_with_score(
        query,
        k=k,
    )


def get_document_count() -> int:
    """
    Return the number of documents currently stored
    in the ChromaDB collection.
    """

    vector_store = get_vector_store()

    collection = vector_store._collection

    return collection.count()


def clear_vector_store() -> None:
    """
    Delete all documents from the current ChromaDB collection.

    This function is intended for controlled application
    reset operations.
    """

    vector_store = get_vector_store()

    collection = vector_store._collection

    existing = collection.get()

    ids = existing.get("ids", [])

    if ids:
        collection.delete(ids=ids)