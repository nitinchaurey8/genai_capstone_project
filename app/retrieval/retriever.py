from langchain_core.documents import Document

from app.vectorstore.chroma_store import (
    similarity_search_with_scores,
)


# Number of chunks retrieved by default.
DEFAULT_TOP_K = 4


def retrieve_documents(
    query: str,
    k: int = DEFAULT_TOP_K,
) -> list[Document]:
    """
    Retrieve the most relevant document chunks
    for a user question.
    """

    if not query or not query.strip():
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    if k <= 0:
        raise ValueError(
            "Number of retrieved documents must "
            "be greater than zero."
        )

    results = similarity_search_with_scores(
        query=query,
        k=k,
    )

    return [
        document
        for document, _score in results
    ]


def retrieve_documents_with_scores(
    query: str,
    k: int = DEFAULT_TOP_K,
) -> list[tuple[Document, float]]:
    """
    Retrieve relevant document chunks together
    with their similarity scores.
    """

    if not query or not query.strip():
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    if k <= 0:
        raise ValueError(
            "Number of retrieved documents must "
            "be greater than zero."
        )

    return similarity_search_with_scores(
        query=query,
        k=k,
    )


def build_context(
    documents: list[Document],
) -> str:
    """
    Combine retrieved document chunks into a single
    context string for the RAG generation step.
    """

    if not documents:
        return ""

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get("page")
        sheet = document.metadata.get("sheet")
        row = document.metadata.get("row")

        location_parts = []

        if page is not None:
            location_parts.append(
                f"page {page}"
            )

        if sheet is not None:
            location_parts.append(
                f"sheet {sheet}"
            )

        if row is not None:
            location_parts.append(
                f"row {row}"
            )

        if location_parts:
            source_reference = (
                f"{source} "
                f"({', '.join(location_parts)})"
            )
        else:
            source_reference = source

        context_parts.append(
            f"[Source {index}: {source_reference}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)