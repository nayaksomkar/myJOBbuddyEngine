"""PDF-to-text utility function using the shared root configuration."""

import pymupdf
from pdf2txt_config import pdf_path

def extract_and_print(pdf_file: str = pdf_path) -> None:
    """Print each page from the configured PDF file."""
    doc = pymupdf.open(pdf_file)
    try:
        for page in doc:
            print(page.get_text())
    finally:
        doc.close()


if __name__ == "__main__":
    extract_and_print()