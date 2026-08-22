from pathlib import Path
import tempfile

from openpyxl import Workbook

from app.ingestion.document_loader import load_document


def main():
    print("Starting document loader test...\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # ==========================================
        # TXT TEST
        # ==========================================

        txt_file = temp_path / "test.txt"

        txt_file.write_text(
            "This is a test document for the GenAI Capstone RAG project.",
            encoding="utf-8",
        )

        documents = load_document(str(txt_file))

        assert len(documents) == 1
        assert "GenAI Capstone" in documents[0].page_content
        assert documents[0].metadata["file_type"] == "txt"

        print("TXT loader: PASSED")

        # ==========================================
        # CSV TEST
        # ==========================================

        csv_file = temp_path / "test.csv"

        csv_file.write_text(
            "name,city,plan\n"
            "Amit,Mumbai,Monthly\n"
            "Rahul,Pune,Weekly\n",
            encoding="utf-8",
        )

        documents = load_document(str(csv_file))

        assert len(documents) == 2
        assert documents[0].metadata["file_type"] == "csv"
        assert "Amit" in documents[0].page_content

        print("CSV loader: PASSED")

        # ==========================================
        # XLSX TEST
        # ==========================================

        xlsx_file = temp_path / "test.xlsx"

        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Customers"

        worksheet.append(["name", "city", "plan"])
        worksheet.append(["Amit", "Mumbai", "Monthly"])
        worksheet.append(["Rahul", "Pune", "Weekly"])

        workbook.save(xlsx_file)

        documents = load_document(str(xlsx_file))

        assert len(documents) == 2
        assert documents[0].metadata["file_type"] == "xlsx"
        assert documents[0].metadata["sheet"] == "Customers"
        assert "Amit" in documents[0].page_content

        print("XLSX loader: PASSED")

        # ==========================================
        # UNSUPPORTED FILE TEST
        # ==========================================

        unsupported_file = temp_path / "test.docx"

        unsupported_file.write_text(
            "This file type is not supported.",
            encoding="utf-8",
        )

        try:
            load_document(str(unsupported_file))

            raise AssertionError(
                "Unsupported file type was not rejected."
            )

        except ValueError:
            pass

        print("Unsupported file rejection: PASSED")

    print("\nDOCUMENT LOADER TEST: PASSED")


if __name__ == "__main__":
    main()