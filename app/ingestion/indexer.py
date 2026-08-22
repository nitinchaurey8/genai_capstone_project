"""
Module: Document indexing pipeline for the GenAI RAG application

Purpose:
Processes loaded LangChain Documents into chunks, obtains the configured embedding model, and stores the chunks in ChromaDB. It returns counts and source names describing the indexing operation. It also provides a convenience function for loading and indexing one filesystem file.

Responsibilities:
- Validate that input Documents and generated chunks are available
- Chunk Documents through the processing component
- Verify the embedding service before storage
- Add chunks to ChromaDB and return an indexing summary

Project Role:
This module connects ingestion to the processing, embedding, and vector store layers. The Streamlit application uses it to index uploaded content, while tests and backend integrations can use its single-file helper.

Technologies:
- LangChain Document
- pathlib
- Application chunking and embedding services
- ChromaDB through the application vector store module

Important:
Indexing raises validation errors when no documents, chunks, embedding model, or stored document IDs are available. The returned summary reports loaded documents, created chunks, stored vectors, and distinct source names.
"""

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