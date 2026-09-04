"""Test sending parsed data to the external ChromaDB client."""

import unittest
from unittest.mock import patch

import main


class ParseToChromaTests(unittest.TestCase):
    def test_parse_data_to_chromadb(self):
        collection = unittest.mock.Mock()
        client = unittest.mock.Mock()
        client.get_or_create_collection.return_value = collection
        model = unittest.mock.Mock()
        model.embed_documents.return_value = [[0.1, 0.2]]

        with patch.object(main, "NVIDIAEmbeddings", return_value=model), patch.object(
            main.chromadb, "HttpClient", return_value=client
        ):
            main.store_vectors("test-document", {"name": "Test Candidate"})

        collection.upsert.assert_called_once()
        self.assertEqual(collection.upsert.call_args.kwargs["ids"], ["test-document"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
