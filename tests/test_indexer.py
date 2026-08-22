from pathlib import Path
import tempfile

from app.ingestion.indexer import (
    index_documents,
)
from app.vectorstore.chroma_store import (
    clear_vector_store,
    get_document_count,
    similarity_search,
)
from langchain_core.documents import Document


def main():
    print("Starting document indexer test...\n")

    # ------------------------------------------
    # CLEAN TEST VECTOR STORE
    # ------------------------------------------

    clear_vector_store()

    # ------------------------------------------
    # CREATE TEST DOCUMENTS
    # ------------------------------------------

    documents = [
        Document(
            page_content=(
                "ChromaDB is a vector database used "
                "to store embeddings and perform "
                "similarity search."
            ),
            metadata={
                "source": "indexer_test.txt",
                "file_type": "txt",
            },
        ),
        Document(
            page_content=(
                "LangChain is a framework for developing "
                "applications powered by language models."
            ),
            metadata={
                "source": "indexer_test.txt",
                "file_type": "txt",
            },
        ),
    ]

    print(
        f"Input documents: {len(documents)}"
    )

    # ------------------------------------------
    # INDEX DOCUMENTS
    # ------------------------------------------

    result = index_documents(documents)

    assert result is not None

    assert (
        result["documents_loaded"] == 2
    )

    assert (
        result["chunks_created"] > 0
    )

    assert (
        result["vectors_created"]
        == result["chunks_created"]
    )

    print(
        "Document indexing: PASSED"
    )

    print(
        "Documents loaded:",
        result["documents_loaded"],
    )

    print(
        "Chunks created:",
        result["chunks_created"],
    )

    print(
        "Vectors created:",
        result["vectors_created"],
    )

    # ------------------------------------------
    # VERIFY CHROMADB
    # ------------------------------------------

    stored_count = get_document_count()

    assert (
        stored_count
        == result["vectors_created"]
    )

    print(
        "ChromaDB storage verification: PASSED"
    )

    # ------------------------------------------
    # VERIFY SEARCH
    # ------------------------------------------

    results = similarity_search(
        query="What is ChromaDB used for?",
        k=2,
    )

    assert results

    assert "ChromaDB" in results[0].page_content

    print(
        "Indexed document retrieval: PASSED"
    )

    print("\nRetrieved content:")
    print(
        results[0].page_content
    )

    # ------------------------------------------
    # VERIFY SOURCES
    # ------------------------------------------

    assert (
        "indexer_test.txt"
        in result["sources"]
    )

    print(
        "Source tracking: PASSED"
    )

    # ------------------------------------------
    # EMPTY DOCUMENT TEST
    # ------------------------------------------

    try:
        index_documents([])

        raise AssertionError(
            "Empty document list was not rejected."
        )

    except ValueError:
        pass

    print(
        "Empty document rejection: PASSED"
    )

    # ------------------------------------------
    # CLEANUP
    # ------------------------------------------

    clear_vector_store()

    assert (
        get_document_count() == 0
    )

    print(
        "Indexer test cleanup: PASSED"
    )

    # ------------------------------------------
    # FINAL RESULT
    # ------------------------------------------

    print(
        "\nDOCUMENT INDEXER TEST: PASSED"
    )


if __name__ == "__main__":
    main()