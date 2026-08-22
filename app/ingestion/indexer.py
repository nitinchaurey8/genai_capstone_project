from pathlib import Path

from langchain_core.documents import Document

from app.embeddings.embedding_service import get_embedding_model
from app.processing.chunker import chunk_documents
from app.vectorstore.chroma_store import add_documents


def index_documents(
    documents: list[Document],
) -> dict:
    """
    Process and index loaded documents.

    Flow:

        Loaded Documents
              ↓
           Chunking
              ↓
          Embeddings
              ↓
           ChromaDB

    Returns information about the indexing operation.
    """

    if not documents:
        raise ValueError(
            "No documents were provided for indexing."
        )

    # ------------------------------------------
    # CHUNK DOCUMENTS
    # ------------------------------------------

    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError(
            "Documents could not be converted into chunks."
        )

    # ------------------------------------------
    # VERIFY EMBEDDING SERVICE
    # ------------------------------------------

    embedding_model = get_embedding_model()

    if embedding_model is None:
        raise ValueError(
            "Embedding model could not be initialized."
        )

    # ------------------------------------------
    # STORE CHUNKS IN CHROMADB
    # ------------------------------------------

    document_ids = add_documents(chunks)

    if not document_ids:
        raise ValueError(
            "Documents could not be added to ChromaDB."
        )

    # ------------------------------------------
    # COLLECT SOURCE INFORMATION
    # ------------------------------------------

    sources = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        if source not in sources:
            sources.append(source)

    # ------------------------------------------
    # RETURN INDEXING SUMMARY
    # ------------------------------------------

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "vectors_created": len(document_ids),
        "sources": sources,
    }


def index_uploaded_file(
    file_path: str | Path,
) -> dict:
    """
    Load and index a single file.

    This function is primarily useful for testing
    and backend integration.
    """

    from app.ingestion.document_loader import (
        load_document,
    )

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    documents = load_document(path)

    return index_documents(documents)