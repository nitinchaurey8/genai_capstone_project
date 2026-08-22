from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".csv",
    ".xlsx",
}


def load_document(file_path: str) -> list[Document]:
    """
    Load a supported file and convert it into LangChain Documents.

    Supported formats:
    - PDF
    - TXT
    - CSV
    - XLSX
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        return _load_pdf(path)

    if extension == ".txt":
        return _load_txt(path)

    if extension == ".csv":
        return _load_csv(path)

    if extension == ".xlsx":
        return _load_xlsx(path)

    raise ValueError(f"Unsupported file type: {extension}")


def _load_pdf(path: Path) -> list[Document]:
    """Load text from a PDF file."""

    reader = PdfReader(str(path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "file_type": "pdf",
                        "page": page_number,
                    },
                )
            )

    return documents


def _load_txt(path: Path) -> list[Document]:
    """Load text from a TXT file."""

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "file_type": "txt",
            },
        )
    ]


def _load_csv(path: Path) -> list[Document]:
    """Load a CSV file and convert each row into a Document."""

    dataframe = pd.read_csv(path)

    documents = []

    for row_number, row in dataframe.iterrows():
        row_data = "\n".join(
            f"{column}: {value}"
            for column, value in row.items()
        )

        if row_data.strip():
            documents.append(
                Document(
                    page_content=row_data,
                    metadata={
                        "source": path.name,
                        "file_type": "csv",
                        "row": int(row_number) + 1,
                    },
                )
            )

    return documents


def _load_xlsx(path: Path) -> list[Document]:
    """Load all worksheets from an Excel file."""

    documents = []

    with pd.ExcelFile(path) as excel_file:
        for sheet_name in excel_file.sheet_names:
            dataframe = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
            )

            for row_number, row in dataframe.iterrows():
                row_data = "\n".join(
                    f"{column}: {value}"
                    for column, value in row.items()
                )

                if row_data.strip():
                    documents.append(
                        Document(
                            page_content=row_data,
                            metadata={
                                "source": path.name,
                                "file_type": "xlsx",
                                "sheet": sheet_name,
                                "row": int(row_number) + 1,
                            },
                        )
                    )

    return documents
