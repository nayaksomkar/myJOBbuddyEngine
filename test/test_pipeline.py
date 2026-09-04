"""Pipeline runner: call the production functions in order."""

import tempfile
import unittest
from unittest.mock import patch

import main
from UnwrapPDF.pdf_extract import extract_text
from resumeDataParse.main import parse_text

from .helpers import FakeChain, FakeParser, sample_pdf_bytes


def run_pipeline() -> dict:
    """Run the real production stages with offline test doubles."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temporary_file:
        temporary_file.write(sample_pdf_bytes())
        temporary_file.flush()
        extracted_text = extract_text(temporary_file.name)

    with patch("resumeDataParse.main.build_chain", return_value=(FakeChain(), FakeParser())):
        parsed_data = parse_text(extracted_text)

    collection = unittest.mock.Mock()
    client = unittest.mock.Mock()
    client.get_or_create_collection.return_value = collection
    model = unittest.mock.Mock()
    model.embed_documents.return_value = [[0.1, 0.2]]
    with patch.object(main, "NVIDIAEmbeddings", return_value=model), patch.object(
        main.chromadb, "HttpClient", return_value=client
    ):
        main.store_vectors("test-document", parsed_data)

    return {"status": main.health(), "data": parsed_data}


class PipelineTests(unittest.TestCase):
    """Check that the separate step functions compose successfully."""

    def test_pipeline(self):
        result = run_pipeline()
        self.assertEqual(result["status"], {"status": "ok"})
        self.assertEqual(result["data"]["name"], "Test Candidate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
