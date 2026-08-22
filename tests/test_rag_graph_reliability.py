from langchain_core.documents import Document

from app.graph.rag_graph import (
    build_rag_graph,
    run_rag_graph,
)
from app.reliability.safety import (
    FALLBACK_ANSWER,
)
from app.vectorstore.chroma_store import (
    add_documents,
    clear_vector_store,
)


def main():
    print(
        "Starting reliability-aware LangGraph test...\n"
    )

    # ------------------------------------------
    # TEST SETUP
    # ------------------------------------------

    clear_vector_store()

    test_documents = [
        Document(
            page_content=(
                "ChromaDB is a vector database used to "
                "store embeddings and perform similarity "
                "search."
            ),
            metadata={
                "source": "chromadb.txt",
                "file_type": "txt",
            },
        ),
        Document(
            page_content=(
                "LangChain is a framework for developing "
                "applications powered by language models."
            ),
            metadata={
                "source": "langchain.txt",
                "file_type": "txt",
            },
        ),
        Document(
            page_content=(
                "Streamlit is a Python framework used to "
                "create interactive web applications."
            ),
            metadata={
                "source": "streamlit.txt",
                "file_type": "txt",
            },
        ),
    ]

    add_documents(test_documents)

    print(
        "Test documents added: PASSED"
    )

    # ------------------------------------------
    # TEST 1: GRAPH CREATION
    # ------------------------------------------

    graph = build_rag_graph()

    assert graph is not None

    print(
        "Reliability-aware graph creation: PASSED"
    )

    # ------------------------------------------
    # TEST 2: NORMAL QUESTION
    # ------------------------------------------

    question = "What is ChromaDB used for?"

    result = run_rag_graph(
        question=question,
        top_k=2,
    )

    assert result["status"] == "generated"

    assert result["answer"]

    assert "ChromaDB" in result["answer"]

    print(
        "Normal RAG execution: PASSED"
    )

    print("\nGenerated answer:")
    print(result["answer"])

    # ------------------------------------------
    # TEST 3: SOURCE ATTRIBUTION
    # ------------------------------------------

    assert result["sources"]

    assert (
        result["sources"][0]["source"]
        == "chromadb.txt"
    )

    assert (
        "internal_secret"
        not in result["sources"][0]
    )

    print(
        "Safe source attribution: PASSED"
    )

    # ------------------------------------------
    # TEST 4: RETRIEVAL SCORES
    # ------------------------------------------

    assert result["retrieval_scores"]

    assert all(
        isinstance(score, float)
        for score in result[
            "retrieval_scores"
        ]
    )

    print(
        "Retrieval score tracking: PASSED"
    )

    # ------------------------------------------
    # TEST 5: EMPTY QUESTION
    # ------------------------------------------

    empty_result = run_rag_graph(
        question="",
    )

    assert (
        empty_result["status"]
        == "validation_failed"
    )

    assert empty_result["answer"]

    assert (
        empty_result["documents"] == []
    )

    print(
        "Empty question safety control: PASSED"
    )

    # ------------------------------------------
    # TEST 6: WHITESPACE QUESTION
    # ------------------------------------------

    whitespace_result = run_rag_graph(
        question="   ",
    )

    assert (
        whitespace_result["status"]
        == "validation_failed"
    )

    assert whitespace_result["answer"]

    print(
        "Whitespace question safety control: PASSED"
    )

    # ------------------------------------------
    # TEST 7: TOP-K LIMIT
    # ------------------------------------------

    limited_result = run_rag_graph(
        question="What is ChromaDB?",
        top_k=100,
    )

    assert limited_result["status"] == "generated"

    assert len(
        limited_result["documents"]
    ) <= 8

    print(
        "Maximum top-k safety control: PASSED"
    )

    # ------------------------------------------
    # TEST 8: NO CONTEXT
    # ------------------------------------------

    clear_vector_store()

    no_context_result = run_rag_graph(
        question=(
            "What is the capital of Mars?"
        ),
        top_k=4,
    )

    assert (
        no_context_result["status"]
        == "no_context"
    )

    assert (
        no_context_result["answer"]
        == FALLBACK_ANSWER
    )

    print(
        "No-context fallback: PASSED"
    )

    # ------------------------------------------
    # TEST 9: NO SOURCE LEAKAGE
    # ------------------------------------------

    assert (
        "GOOGLE_API_KEY"
        not in no_context_result["answer"]
    )

    assert (
        "OPENAI_API_KEY"
        not in no_context_result["answer"]
    )

    print(
        "Sensitive configuration protection: PASSED"
    )

    # ------------------------------------------
    # CLEANUP
    # ------------------------------------------

    clear_vector_store()

    print(
        "Reliability test cleanup: PASSED"
    )

    # ------------------------------------------
    # FINAL RESULT
    # ------------------------------------------

    print(
        "\nRELIABILITY-AWARE LANGGRAPH TEST: PASSED"
    )


if __name__ == "__main__":
    main()