from pathlib import Path

from app.ingestion.document_loader import load_document


PDF_PATH = Path("data/test_document.pdf")


def main():
    print("Starting PDF loader test...\n")

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Test PDF not found: {PDF_PATH}\n"
            "Place a PDF file at data/test_document.pdf"
        )

    documents = load_document(str(PDF_PATH))

    if not documents:
        raise RuntimeError(
            "PDF was loaded but no extractable text was found."
        )

    print("PDF loader: PASSED")
    print("Pages loaded:", len(documents))

    first_document = documents[0]

    print("Source:", first_document.metadata.get("source"))
    print("File type:", first_document.metadata.get("file_type"))
    print("First page:", first_document.metadata.get("page"))

    print("\nFirst page preview:")
    print(first_document.page_content[:500])

    print("\nPDF LOADER TEST: PASSED")


if __name__ == "__main__":
    main()