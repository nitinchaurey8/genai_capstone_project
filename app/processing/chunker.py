from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Split loaded LangChain Documents into smaller chunks.

    The original document metadata is preserved on every chunk.
    """

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index metadata so individual chunks
    # can be traced back during retrieval.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks