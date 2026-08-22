import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st

from app.graph.rag_graph import run_rag_graph
from app.ingestion.document_loader import load_document
from app.ingestion.indexer import index_documents
from app.reliability.safety import validate_file_extension
from app.reliability.safety import validate_question
from app.vectorstore.chroma_store import clear_vector_store

from app.ingestion.document_loader import (
    load_document,
)
from app.ingestion.indexer import (
    index_documents,
)
from app.reliability.safety import (
    validate_file_extension,
    validate_question,
)
from app.vectorstore.chroma_store import (
    clear_vector_store,
)


# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="GenAI Capstone RAG",
    page_icon="📚",
    layout="wide",
)


# ------------------------------------------
# PAGE HEADER
# ------------------------------------------

st.title(
    "GenAI Capstone Project"
)

st.subheader(
    "Retrieval-Augmented Generation Application"
)

st.write(
    "Upload documents and ask questions using "
    "a grounded RAG workflow."
)


# ------------------------------------------
# SESSION STATE
# ------------------------------------------

if "documents_indexed" not in st.session_state:
    st.session_state.documents_indexed = False

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "indexing_summary" not in st.session_state:
    st.session_state.indexing_summary = None


# ------------------------------------------
# SIDEBAR
# ------------------------------------------

with st.sidebar:

    st.header("Document Upload")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "txt",
            "csv",
            "xlsx",
        ],
        accept_multiple_files=True,
    )

    process_documents = st.button(
        "Process Documents",
        type="primary",
        use_container_width=True,
    )

    clear_documents = st.button(
        "Clear Documents",
        use_container_width=True,
    )


# ------------------------------------------
# CLEAR DOCUMENTS
# ------------------------------------------

if clear_documents:

    try:
        clear_vector_store()

        st.session_state.documents_indexed = False
        st.session_state.indexed_files = []
        st.session_state.indexing_summary = None

        st.success(
            "Documents cleared successfully."
        )

    except Exception:
        st.error(
            "Unable to clear the document collection."
        )


# ------------------------------------------
# PROCESS DOCUMENTS
# ------------------------------------------

if process_documents:

    if not uploaded_files:

        st.warning(
            "Please upload at least one document."
        )

    else:

        loaded_documents = []
        file_names = []

        progress = st.progress(0)

        try:

            total_files = len(uploaded_files)

            for index, uploaded_file in enumerate(
                uploaded_files,
                start=1,
            ):

                file_name = uploaded_file.name

                valid, message = (
                    validate_file_extension(
                        file_name
                    )
                )

                if not valid:
                    st.error(message)
                    continue

                # Create a temporary file in memory-compatible
                # Streamlit processing.
                import tempfile
                from pathlib import Path

                suffix = Path(
                    file_name
                ).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temporary_file:

                    temporary_file.write(
                        uploaded_file.getvalue()
                    )

                    temporary_path = (
                        temporary_file.name
                    )

                documents = load_document(
                    temporary_path
                )

                # Replace the temporary path in
                # metadata with the original filename.
                for document in documents:

                    document.metadata[
                        "source"
                    ] = file_name

                loaded_documents.extend(
                    documents
                )

                file_names.append(
                    file_name
                )

                progress.progress(
                    index / total_files
                )

            if not loaded_documents:

                st.error(
                    "No valid documents could be loaded."
                )

            else:

                summary = index_documents(
                    loaded_documents
                )

                st.session_state.documents_indexed = True

                st.session_state.indexed_files = (
                    file_names
                )

                st.session_state.indexing_summary = (
                    summary
                )

                st.success(
                    "Documents processed successfully."
                )

        except Exception:

            st.error(
                "Something went wrong while processing "
                "the uploaded documents. Please try again."
            )


# ------------------------------------------
# INDEXING SUMMARY
# ------------------------------------------

if st.session_state.indexing_summary:

    summary = (
        st.session_state.indexing_summary
    )

    st.subheader(
        "Document Processing Summary"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Documents",
        summary["documents_loaded"],
    )

    col2.metric(
        "Chunks",
        summary["chunks_created"],
    )

    col3.metric(
        "Vectors",
        summary["vectors_created"],
    )

    st.write("Indexed files:")

    for file_name in (
        st.session_state.indexed_files
    ):
        st.write(
            f"- {file_name}"
        )


# ------------------------------------------
# QUESTION AREA
# ------------------------------------------

st.divider()

st.header(
    "Ask a Question"
)

question = st.text_input(
    "Enter your question",
    placeholder=(
        "Ask something about your uploaded documents..."
    ),
)

ask_question = st.button(
    "Ask Question",
    type="primary",
)


# ------------------------------------------
# QUESTION PROCESSING
# ------------------------------------------

if ask_question:

    valid, message = validate_question(
        question
    )

    if not valid:

        st.warning(message)

    elif not st.session_state.documents_indexed:

        st.warning(
            "Please upload and process documents "
            "before asking a question."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                result = run_rag_graph(
                    question=question
                )

                # ----------------------------------
                # ERROR
                # ----------------------------------

                if result.get("error"):

                    st.error(
                        result["error"]
                    )

                # ----------------------------------
                # ANSWER
                # ----------------------------------

                else:

                    st.subheader(
                        "Generated Answer"
                    )

                    st.write(
                        result.get(
                            "answer",
                            "",
                        )
                    )

                    # ----------------------------------
                    # SOURCES
                    # ----------------------------------

                    sources = result.get(
                        "sources",
                        [],
                    )

                    if sources:

                        st.subheader(
                            "Sources"
                        )

                        for source in sources:

                            source_name = source.get(
                                "source",
                                "Unknown source",
                            )

                            file_type = source.get(
                                "file_type"
                            )

                            page = source.get(
                                "page"
                            )

                            details = (
                                f"- {source_name}"
                            )

                            if file_type:
                                details += (
                                    f" ({file_type})"
                                )

                            if page is not None:
                                details += (
                                    f" — page {page}"
                                )

                            st.write(details)

                    else:

                        st.info(
                            "No source information "
                            "was available."
                        )

            except Exception:

                st.error(
                    "Something went wrong while "
                    "processing your question."
                )