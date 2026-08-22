from langchain_core.documents import Document

from app.processing.chunker import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_documents,
)


def main():
    print("Starting document chunker test...\n")

    # Create a document that is intentionally longer
    # than the configured chunk size.
    paragraphs = [
        (
            "The GenAI Capstone Project is a Retrieval-Augmented "
            "Generation application. The application allows users "
            "to upload different types of documents."
        ),
        (
            "Supported documents include PDF, TXT, CSV, and Excel "
            "files. The uploaded documents are loaded and converted "
            "into LangChain Document objects."
        ),
        (
            "The documents are then divided into smaller chunks. "
            "Each chunk is converted into an embedding vector. "
            "The vectors are stored in ChromaDB."
        ),
        (
            "When the user asks a question, the question is also "
            "converted into an embedding. ChromaDB performs "
            "similarity search to identify the most relevant "
            "document chunks."
        ),
        (
            "The retrieved content is then provided to the language "
            "model as context. The language model generates the "
            "final answer."
        ),
        (
            "The application uses LangChain for document processing "
            "and retrieval. LangGraph is used to define the "
            "application workflow and connect the RAG processing "
            "steps."
        ),
        (
            "Streamlit provides the frontend interface. Users can "
            "upload documents, submit questions, and view the "
            "generated answers through the application."
        ),
        (
            "The RAG pipeline consists of document ingestion, "
            "chunking, embedding generation, vector storage, "
            "similarity search, context retrieval, prompt "
            "construction, and answer generation."
        ),
    ]

    text = "\n\n".join(paragraphs)

    print("Original document length:", len(text))
    print("Configured chunk size:", CHUNK_SIZE)
    print("Configured chunk overlap:", CHUNK_OVERLAP)

    # Create the LangChain Document.
    document = Document(
        page_content=text,
        metadata={
            "source": "test_document.txt",
            "file_type": "txt",
        },
    )

    # Run the chunker.
    chunks = chunk_documents([document])

    # ------------------------------------------
    # BASIC VALIDATION
    # ------------------------------------------

    assert chunks, "Chunker returned no chunks."

    assert len(chunks) > 1, (
        "Test document was not split into multiple chunks."
    )

    # ------------------------------------------
    # CHUNK SIZE VALIDATION
    # ------------------------------------------

    for chunk in chunks:
        assert len(chunk.page_content) <= CHUNK_SIZE, (
            f"Chunk exceeds configured size: "
            f"{len(chunk.page_content)} > {CHUNK_SIZE}"
        )

    # ------------------------------------------
    # METADATA VALIDATION
    # ------------------------------------------

    for chunk in chunks:
        assert chunk.metadata["source"] == "test_document.txt"
        assert chunk.metadata["file_type"] == "txt"
        assert "chunk_id" in chunk.metadata

    # ------------------------------------------
    # CHUNK ID VALIDATION
    # ------------------------------------------

    chunk_ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    expected_ids = list(range(len(chunks)))

    assert chunk_ids == expected_ids, (
        "Chunk IDs are not sequential."
    )

    # ------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------

    print("Number of chunks:", len(chunks))

    print("\nChunks:")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}")
        print("Chunk ID:", chunk.metadata["chunk_id"])
        print("Length:", len(chunk.page_content))
        print("Source:", chunk.metadata["source"])
        print("File type:", chunk.metadata["file_type"])
        print("Content preview:")
        print(chunk.page_content[:250])

    print("\nDOCUMENT CHUNKER TEST: PASSED")


if __name__ == "__main__":
    main()