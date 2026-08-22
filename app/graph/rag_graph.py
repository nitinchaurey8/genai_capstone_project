from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.rag.rag_chain import generate_answer
from app.reliability.safety import (
    DEFAULT_TOP_K,
    FALLBACK_ANSWER,
    build_safe_source,
    has_retrieved_context,
    normalize_top_k,
    sanitize_error_message,
    validate_question,
)
from app.retrieval.retriever import (
    build_context,
    retrieve_documents_with_scores,
)


class RAGState(TypedDict, total=False):
    """
    State passed between LangGraph nodes.
    """

    question: str
    top_k: int
    documents: list
    context: str
    answer: str
    sources: list
    retrieval_scores: list
    error: str
    status: str


def validate_node(state: RAGState) -> RAGState:
    """
    Validate the user's question before starting
    the RAG process.
    """

    question = state.get("question", "")

    valid, message = validate_question(question)

    if not valid:
        return {
            **state,
            "question": question,
            "documents": [],
            "context": "",
            "answer": message,
            "sources": [],
            "retrieval_scores": [],
            "error": "",
            "status": "validation_failed",
        }

    top_k = normalize_top_k(
        state.get("top_k", DEFAULT_TOP_K)
    )

    return {
        **state,
        "question": question.strip(),
        "top_k": top_k,
        "status": "validated",
        "error": "",
    }


def retrieve_node(state: RAGState) -> RAGState:
    """
    Retrieve relevant document chunks.
    """

    if state.get("status") == "validation_failed":
        return state

    question = state.get("question", "")
    top_k = state.get(
        "top_k",
        DEFAULT_TOP_K,
    )

    try:
        results = retrieve_documents_with_scores(
            query=question,
            k=top_k,
        )

        documents = [
            document
            for document, _score in results
        ]

        scores = [
            score
            for _document, score in results
        ]

        return {
            **state,
            "documents": documents,
            "retrieval_scores": scores,
            "error": "",
            "status": "retrieved",
        }

    except Exception as exc:
        return {
            **state,
            "documents": [],
            "retrieval_scores": [],
            "error": sanitize_error_message(exc),
            "status": "retrieval_failed",
        }


def context_node(state: RAGState) -> RAGState:
    """
    Build the context from retrieved documents.
    """

    if state.get("status") in {
        "validation_failed",
        "retrieval_failed",
    }:
        return state

    documents = state.get(
        "documents",
        [],
    )

    if not has_retrieved_context(documents):
        return {
            **state,
            "context": "",
            "answer": FALLBACK_ANSWER,
            "status": "no_context",
        }

    context = build_context(documents)

    if not context.strip():
        return {
            **state,
            "context": "",
            "answer": FALLBACK_ANSWER,
            "status": "no_context",
        }

    return {
        **state,
        "context": context,
        "status": "context_ready",
        "error": "",
    }


def generate_node(state: RAGState) -> RAGState:
    """
    Generate a grounded answer using only the
    retrieved documents.
    """

    if state.get("status") in {
        "validation_failed",
        "retrieval_failed",
        "no_context",
    }:
        return state

    question = state.get(
        "question",
        "",
    )

    documents = state.get(
        "documents",
        [],
    )

    try:
        answer = generate_answer(
            question=question,
            documents=documents,
        )

        return {
            **state,
            "answer": answer,
            "status": "generated",
            "error": "",
        }

    except Exception as exc:
        return {
            **state,
            "answer": sanitize_error_message(exc),
            "error": sanitize_error_message(exc),
            "status": "generation_failed",
        }


def source_node(state: RAGState) -> RAGState:
    """
    Build safe source metadata for frontend display.
    """

    documents = state.get(
        "documents",
        [],
    )

    sources = [
        build_safe_source(document)
        for document in documents
    ]

    return {
        **state,
        "sources": sources,
    }


def build_rag_graph():
    """
    Build and compile the reliability-aware
    LangGraph RAG workflow.

    Flow:

        START
          ↓
       Validate
          ↓
       Retrieve
          ↓
       Context
          ↓
       Generate
          ↓
       Sources
          ↓
         END
    """

    graph = StateGraph(RAGState)

    graph.add_node(
        "validate",
        validate_node,
    )

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "context",
        context_node,
    )

    graph.add_node(
        "generate",
        generate_node,
    )

    graph.add_node(
        "sources",
        source_node,
    )

    graph.add_edge(
        START,
        "validate",
    )

    graph.add_edge(
        "validate",
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "context",
    )

    graph.add_edge(
        "context",
        "generate",
    )

    graph.add_edge(
        "generate",
        "sources",
    )

    graph.add_edge(
        "sources",
        END,
    )

    return graph.compile()


def run_rag_graph(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> RAGState:
    """
    Execute the complete reliability-aware
    LangGraph RAG workflow.
    """

    graph = build_rag_graph()

    initial_state: RAGState = {
        "question": question,
        "top_k": top_k,
        "documents": [],
        "context": "",
        "answer": "",
        "sources": [],
        "retrieval_scores": [],
        "error": "",
        "status": "started",
    }

    return graph.invoke(initial_state)