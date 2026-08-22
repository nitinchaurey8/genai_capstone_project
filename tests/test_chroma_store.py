from app.vectorstore.chroma_store import (
    add_documents,
    clear_vector_store,
    get_document_count,
    similarity_search_with_scores,
)
from langchain_core.documents import Document


def main():
    print("Starting ChromaDB vector store test...\n")

    # --------------------------------------------------
    # 1. Clear previous test data
    # --------------------------------------------------

    clear_vector_store()

    initial_count = get_document_count()

    assert initial_count == 0, (
        f"Expected empty vector store, "
        f"but found {initial_count} documents."
    )

    print("Initial vector store: EMPTY")

    # --------------------------------------------------
    # 2. Create test documents
    # --------------------------------------------------

    documents = [
        Document(
            page_content=(
                "LangChain is a framework for building "
                "applications powered by language models."
            ),
            metadata={
                "source": "langchain.txt",
                "file_type": "txt",
            },
        ),
        Document(
            page_content=(
                "ChromaDB is a vector database used to store "
                "embeddings and perform similarity search."
            ),
            metadata={
                "source": "chromadb.txt",
                "file_type": "txt",
            },
        ),
        Document(
            page_content=(
                "Streamlit is a Python framework for creating "
                "interactive web application interfaces."
            ),
            metadata={
                "source": "streamlit.txt",
                "file_type": "txt",
            },
        ),
    ]

    # --------------------------------------------------
    # 3. Add documents
    # --------------------------------------------------

    document_ids = add_documents(documents)

    assert len(document_ids) == len(documents), (
        "Number of returned document IDs does not "
        "match number of documents added."
    )

    print(
        f"Documents added: {len(document_ids)}"
    )

    # --------------------------------------------------
    # 4. Verify document count
    # --------------------------------------------------

    document_count = get_document_count()

    assert document_count == len(documents), (
        f"Expected {len(documents)} documents, "
        f"but found {document_count}."
    )

    print(
        f"Document count verification: PASSED "
        f"({document_count} documents)"
    )

    # --------------------------------------------------
    # 5. Perform similarity search
    # --------------------------------------------------

    question = "What is ChromaDB used for?"

    results = similarity_search_with_scores(
        question,
        k=2,
    )

    assert results, (
        "ChromaDB returned no search results."
    )

    print("\nSearch question:")
    print(question)

    # --------------------------------------------------
    # 6. Display search results
    # --------------------------------------------------

    print("\nRetrieved results:")

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\nResult {index}")
        print("Score:", score)
        print(
            "Source:",
            document.metadata.get("source"),
        )
        print(
            "File type:",
            document.metadata.get("file_type"),
        )
        print(
            "Content:",
            document.page_content,
        )

    # --------------------------------------------------
    # 7. Verify correct document was retrieved
    # --------------------------------------------------

    best_document = results[0][0]

    assert (
        best_document.metadata.get("source")
        == "chromadb.txt"
    ), (
        "Expected ChromaDB document to be the "
        "top similarity-search result."
    )

    print(
        "\nTop result verification: PASSED"
    )

    # --------------------------------------------------
    # 8. Test empty query handling
    # --------------------------------------------------

    try:
        similarity_search_with_scores(
            "",
            k=2,
        )

        raise AssertionError(
            "Empty query was not rejected."
        )

    except ValueError:
        pass

    print(
        "Empty query rejection: PASSED"
    )

    # --------------------------------------------------
    # 9. Test invalid k handling
    # --------------------------------------------------

    try:
        similarity_search_with_scores(
            "What is ChromaDB?",
            k=0,
        )

        raise AssertionError(
            "Invalid k value was not rejected."
        )

    except ValueError:
        pass

    print(
        "Invalid result count rejection: PASSED"
    )

    # --------------------------------------------------
    # 10. Clean up test data
    # --------------------------------------------------

    clear_vector_store()

    final_count = get_document_count()

    assert final_count == 0, (
        f"Expected vector store to be empty after "
        f"cleanup, but found {final_count} documents."
    )

    print(
        "Vector store cleanup: PASSED"
    )

    print(
        "\nCHROMADB VECTOR STORE TEST: PASSED"
    )


if __name__ == "__main__":
    main()