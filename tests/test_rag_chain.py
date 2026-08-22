from langchain_core.documents import Document

from app.rag.rag_chain import (
    answer_question,
    generate_answer,
    get_llm,
)
from app.vectorstore.chroma_store import (
    add_documents,
    clear_vector_store,
)


def main():
    print("Starting RAG chain test...\n")

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

    print("RAG test documents added: PASSED")

    # ------------------------------------------
    # TEST 1: LLM CONNECTION
    # ------------------------------------------

    llm = get_llm()

    assert llm is not None

    print("RAG LLM creation: PASSED")

    # ------------------------------------------
    # TEST 2: DIRECT GROUNDED ANSWER
    # ------------------------------------------

    question = "What is ChromaDB used for?"

    retrieved_document = test_documents[0]

    answer = generate_answer(
        question=question,
        documents=[retrieved_document],
    )

    assert answer.strip()

    assert "ChromaDB" in answer

    print("Grounded answer generation: PASSED")

    print("\nGenerated answer:")
    print(answer)

    # ------------------------------------------
    # TEST 3: COMPLETE RAG FLOW
    # ------------------------------------------

    result = answer_question(
        question=question,
        k=2,
    )

    assert isinstance(result, dict)

    assert result["question"] == question

    assert result["answer"].strip()

    assert result["sources"]

    assert (
        result["sources"][0]["source"]
        == "chromadb.txt"
    )

    print(
        "\nComplete RAG flow: PASSED"
    )

    print("\nRAG answer:")
    print(result["answer"])

    print("\nRetrieved sources:")

    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"({source['file_type']})"
        )

    # ------------------------------------------
    # TEST 4: EMPTY QUESTION
    # ------------------------------------------

    try:
        answer_question(
            question="",
            k=2,
        )

        raise AssertionError(
            "Empty question was not rejected."
        )

    except ValueError:
        pass

    print(
        "\nEmpty question rejection: PASSED"
    )

    # ------------------------------------------
    # TEST 5: EMPTY DOCUMENT CONTEXT
    # ------------------------------------------

    fallback_answer = generate_answer(
        question="What is ChromaDB?",
        documents=[],
    )

    assert (
        "don't have enough information"
        in fallback_answer
    )

    print(
        "Empty context fallback: PASSED"
    )

    # ------------------------------------------
    # CLEANUP
    # ------------------------------------------

    clear_vector_store()

    print(
        "RAG test vector store cleanup: PASSED"
    )

    # ------------------------------------------
    # FINAL RESULT
    # ------------------------------------------

    print("\nRAG CHAIN TEST: PASSED")


if __name__ == "__main__":
    main()