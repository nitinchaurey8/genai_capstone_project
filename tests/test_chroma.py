from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.utils.config import (
    GOOGLE_API_KEY,
    GOOGLE_EMBEDDING_MODEL,
)


def main():
    print("Starting ChromaDB test...")

    # 1. Create the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    # 2. Create small test documents
    documents = [
        Document(
            page_content=(
                "LangChain is a framework for developing "
                "applications powered by language models."
            ),
            metadata={"source": "test_document_1.txt"},
        ),
        Document(
            page_content=(
                "ChromaDB is a vector database used to store "
                "embeddings and perform similarity search."
            ),
            metadata={"source": "test_document_2.txt"},
        ),
        Document(
            page_content=(
                "Streamlit allows Python developers to create "
                "interactive web applications."
            ),
            metadata={"source": "test_document_3.txt"},
        ),
    ]

    # 3. Create a local ChromaDB collection
    vector_store = Chroma(
        collection_name="capstone_test",
        embedding_function=embeddings,
        persist_directory="data/chroma/test",
    )

    # 4. Add the documents to ChromaDB
    vector_store.add_documents(documents)

    print("Documents added to ChromaDB: OK")

    # 5. Perform similarity search
    question = "What is ChromaDB used for?"

    results = vector_store.similarity_search_with_score(
        question,
        k=2,
    )

    print("\nQuestion:")
    print(question)

    print("\nRetrieved results:")

    for i, (document, score) in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Source:", document.metadata.get("source"))
        print("Score:", score)
        print("Content:", document.page_content)

    # 6. Basic validation
    if not results:
        raise RuntimeError("ChromaDB returned no results.")

    best_document = results[0][0]

    if "ChromaDB" not in best_document.page_content:
        raise RuntimeError(
            "ChromaDB retrieval test failed: "
            "the expected document was not retrieved first."
        )

    print("\nCHROMADB TEST: PASSED")


if __name__ == "__main__":
    main()