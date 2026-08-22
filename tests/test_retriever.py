from langchain_core.documents import Document

from app.retrieval.retriever import (
    DEFAULT_TOP_K,
    build_context,
    retrieve_documents,
    retrieve_documents_with_scores,
)
from app.vectorstore.chroma_store import (
    add_documents,
    clear_vector_store,
)


def main():
    print("Starting retriever test...\n")

    # ------------------------------------------
    # TEST SETUP
    # ------------------------------------------

    clear_vector_store()

    test_documents = [
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

    add_documents(test_documents)

    print("Test documents added: PASSED")

    # ------------------------------------------
    # TEST 1: RETRIEVE DOCUMENTS
    # ------------------------------------------

    query = "What is ChromaDB used for?"

    documents = retrieve_documents(
        query=query,
        k=2,
    )

    assert documents, (
        "Retriever returned no documents."
    )

    assert len(documents) <= 2, (
        "Retriever returned more documents "
        "than requested."
    )

    print("Document retrieval: PASSED")
    print("Documents retrieved:", len(documents))

    # ------------------------------------------
    # TEST 2: VERIFY TOP RESULT
    # ------------------------------------------

    top_document = documents[0]

    assert (
        top_document.metadata.get("source")
        == "chromadb.txt"
    ), (
        "Expected chromadb.txt to be the "
        "top retrieved document."
    )

    print("Top result verification: PASSED")
    print(
        "Top source:",
        top_document.metadata.get("source"),
    )

    # ------------------------------------------
    # TEST 3: RETRIEVAL WITH SCORES
    # ------------------------------------------

    scored_results = retrieve_documents_with_scores(
        query=query,
        k=2,
    )

    assert scored_results, (
        "Scored retrieval returned no results."
    )

    assert len(scored_results) <= 2

    for document, score in scored_results:
        assert document is not None
        assert isinstance(score, float)

    print(
        "Retrieval with scores: PASSED"
    )

    # ------------------------------------------
    # TEST 4: BUILD RAG CONTEXT
    # ------------------------------------------

    context = build_context(documents)

    assert context.strip(), (
        "Context builder returned empty context."
    )

    assert "ChromaDB" in context
    assert "chromadb.txt" in context

    print("Context building: PASSED")

    print("\nGenerated context preview:")
    print(context[:500])

    # ------------------------------------------
    # TEST 5: EMPTY QUERY
    # ------------------------------------------

    try:
        retrieve_documents(
            query="",
            k=2,
        )

        raise AssertionError(
            "Empty query was not rejected."
        )

    except ValueError:
        pass

    print("Empty query rejection: PASSED")

    # ------------------------------------------
    # TEST 6: INVALID K
    # ------------------------------------------

    try:
        retrieve_documents(
            query="What is ChromaDB?",
            k=0,
        )

        raise AssertionError(
            "Invalid k value was not rejected."
        )

    except ValueError:
        pass

    print("Invalid k rejection: PASSED")

    # ------------------------------------------
    # TEST 7: DEFAULT TOP K
    # ------------------------------------------

    assert DEFAULT_TOP_K == 4

    print(
        "Default top-k configuration: PASSED"
    )

    print(
        "Default top-k:",
        DEFAULT_TOP_K,
    )

    # ------------------------------------------
    # CLEANUP
    # ------------------------------------------

    clear_vector_store()

    print("Test vector store cleanup: PASSED")

    # ------------------------------------------
    # FINAL RESULT
    # ------------------------------------------

    print("\nRETRIEVER TEST: PASSED")


if __name__ == "__main__":
    main()