"""Standalone copy of the PDF page-text extraction helper."""

import sys  # for reading command-line arguments
import pymupdf  # PyMuPDF – the library that reads PDF files


def extract_text(pdf_path: str) -> str:
    """
    Opens a PDF, extracts text from every page,
    and labels each page with a '--- Page N ---' header.
    """
    doc = pymupdf.open(pdf_path)       # open the PDF file
    pages = []                         # will hold text of each page
    for i, page in enumerate(doc, start=1):  # loop through pages (1-indexed)
        text = page.get_text()                # extract raw text from the page
        pages.append(f"--- Page {i} ---\n{text}")  # prepend a page marker
    doc.close()                          # close the file
    return "\n\n".join(pages)            # join pages with a blank line


def save_text(text: str, output_path: str) -> None:
    """Writes extracted text to a UTF-8 .txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    """
    CLI entry point.
    Usage: python pdf_extract.py <input.pdf> <output.txt>
    """
    # check that both arguments were provided
    if len(sys.argv) != 3:
        print("Usage: python pdf_extract.py <input.pdf> <output.txt>")
        sys.exit(1)

    pdf_path = sys.argv[1]     # first  argument: path to the input PDF
    output_path = sys.argv[2]  # second argument: path for the output .txt

    text = extract_text(pdf_path)   # step 1: read + extract
    save_text(text, output_path)     # step 2: write to disk
    print(f"Extracted text saved to {output_path}")


if __name__ == "__main__":
    main()
