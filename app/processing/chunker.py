"""
Module: LangChain document chunking utilities

Purpose:
Splits loaded LangChain Documents into smaller overlapping text chunks for downstream embedding and retrieval. It preserves the original metadata on each generated chunk and assigns a sequential chunk identifier.

Responsibilities:
- Handle empty document input
- Configure recursive character-based splitting
- Preserve source and format-specific metadata
- Add chunk identifiers to generated Documents

Project Role:
This module is the processing step between document ingestion and indexing. The indexer uses it before embedding chunks and adding them to the ChromaDB vector store.

Technologies:
- LangChain Document
- LangChain RecursiveCharacterTextSplitter

Important:
The configured chunk size is 1000 characters with 200 characters of overlap. Splitting prioritizes paragraph, line, sentence, space, and character boundaries in that order.
"""

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