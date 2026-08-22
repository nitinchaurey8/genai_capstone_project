from app.embeddings.embedding_service import (
    embed_documents,
    embed_query,
    get_embedding_model,
)
from app.utils.config import GOOGLE_EMBEDDING_MODEL


def main():
    print("Starting embedding service test...\n")

    # ==========================================
    # TEST 1: CREATE EMBEDDING MODEL
    # ==========================================

    embedding_model = get_embedding_model()

    assert embedding_model is not None

    print("Embedding model creation: PASSED")
    print("Embedding model:", GOOGLE_EMBEDDING_MODEL)

    # ==========================================
    # TEST 2: SINGLE QUERY EMBEDDING
    # ==========================================

    query = "What is Retrieval-Augmented Generation?"

    query_vector = embed_query(query)

    assert isinstance(query_vector, list)
    assert len(query_vector) > 0

    assert all(
        isinstance(value, float)
        for value in query_vector
    )

    print("Query embedding: PASSED")
    print("Query vector length:", len(query_vector))

    # ==========================================
    # TEST 3: MULTIPLE DOCUMENT EMBEDDINGS
    # ==========================================

    documents = [
        (
            "LangChain is a framework for developing "
            "applications powered by language models."
        ),
        (
            "ChromaDB is a vector database used to store "
            "embeddings and perform similarity search."
        ),
        (
            "Streamlit is used to create interactive "
            "Python web applications."
        ),
    ]

    document_vectors = embed_documents(documents)

    assert isinstance(document_vectors, list)
    assert len(document_vectors) == len(documents)

    assert all(
        isinstance(vector, list)
        for vector in document_vectors
    )

    assert all(
        len(vector) == len(query_vector)
        for vector in document_vectors
    )

    print("Document embeddings: PASSED")
    print("Documents embedded:", len(document_vectors))
    print(
        "Document vector length:",
        len(document_vectors[0]),
    )

    # ==========================================
    # TEST 4: EMPTY INPUT HANDLING
    # ==========================================

    empty_documents = embed_documents([])

    assert empty_documents == []

    print("Empty document list handling: PASSED")

    try:
        embed_query("")

        raise AssertionError(
            "Empty query was not rejected."
        )

    except ValueError:
        pass

    print("Empty query rejection: PASSED")

    # ==========================================
    # FINAL RESULT
    # ==========================================

    print("\nEMBEDDING SERVICE TEST: PASSED")


if __name__ == "__main__":
    main()