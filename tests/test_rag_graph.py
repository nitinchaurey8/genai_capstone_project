from langchain_core.documents import Document

from app.graph.rag_graph import (
    build_rag_graph,
    run_rag_graph,
)
from app.vectorstore.chroma_store import (
    add_documents,
    clear_vector_store,
)


def main():
    print("Starting LangGraph RAG test...\n")

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

    print("Test documents added: PASSED")

    # ------------------------------------------
    # TEST 1: GRAPH CREATION
    # ------------------------------------------

    graph = build_rag_graph()

    assert graph is not None

    print("LangGraph creation: PASSED")

    # ------------------------------------------
    # TEST 2: COMPLETE GRAPH EXECUTION
    # ------------------------------------------

    question = "What is ChromaDB used for?"

    result = run_rag_graph(question)

    assert result is not None

    assert result["question"] == question

    print("Graph execution: PASSED")

    # ------------------------------------------
    # TEST 3: RETRIEVAL
    # ------------------------------------------

    assert result["documents"]

    print(
        "Graph retrieval: PASSED"
    )

    print(
        "Documents retrieved:",
        len(result["documents"]),
    )

    # ------------------------------------------
    # TEST 4: CONTEXT
    # ------------------------------------------

    assert result["context"]

    assert "ChromaDB" in result["context"]

    print("Graph context creation: PASSED")

    # ------------------------------------------
    # TEST 5: ANSWER
    # ------------------------------------------

    assert result["answer"]

    assert "ChromaDB" in result["answer"]

    print("Graph answer generation: PASSED")

    print("\nGenerated answer:")
    print(result["answer"])

    # ------------------------------------------
    # TEST 6: SOURCES
    # ------------------------------------------

    assert result["sources"]

    assert (
        result["sources"][0]["source"]
        == "chromadb.txt"
    )

    print("Graph source generation: PASSED")

    print("\nSources:")

    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"({source['file_type']})"
        )

    # ------------------------------------------
    # TEST 7: RETRIEVAL SCORES
    # ------------------------------------------

    assert result["retrieval_scores"]

    assert all(
        isinstance(score, float)
        for score in result["retrieval_scores"]
    )

    print(
        "Retrieval score tracking: PASSED"
    )

    # ------------------------------------------
    # TEST 8: EMPTY QUESTION
    # ------------------------------------------

    empty_question_result = run_rag_graph("")

    assert (
        empty_question_result["status"]
        == "validation_failed"
    )

    assert empty_question_result["answer"]

    print(
        "Empty question rejection: PASSED"
    )

    # ------------------------------------------
    # CLEANUP
    # ------------------------------------------

    clear_vector_store()

    print(
        "Graph test vector store cleanup: PASSED"
    )

    # ------------------------------------------
    # FINAL RESULT
    # ------------------------------------------

    print("\nLANGGRAPH RAG TEST: PASSED")


if __name__ == "__main__":
    main()