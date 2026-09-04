"""Step 1: test PDF to text extraction."""

import tempfile
import unittest

from UnwrapPDF.pdf_extract import extract_text
from .helpers import sample_pdf_bytes


class PdfToTxtTests(unittest.TestCase):
    """Check that PDF extraction returns useful page text."""

    def test_pdf_to_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temporary_file:
            temporary_file.write(sample_pdf_bytes())
            temporary_file.flush()
            text = extract_text(temporary_file.name)
        self.assertIn("--- Page 1 ---", text)
        self.assertIn("Test Candidate", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
