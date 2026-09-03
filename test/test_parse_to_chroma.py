"""Step 3: test parsed data to ChromaDB storage."""

import unittest
from unittest.mock import patch

import main


class ParseToChromaTests(unittest.TestCase):
    """Check that parsed data is sent to ChromaDB."""

    def test_parse_data_to_chromadb(self):
        collection = unittest.mock.Mock()
        client = unittest.mock.Mock()
        client.get_or_create_collection.return_value = collection
        model = unittest.mock.Mock()
        model.encode.return_value.tolist.return_value = [[0.1, 0.2]]

        with patch.object(main, "SentenceTransformer", return_value=model), patch.object(
            main.chromadb, "PersistentClient", return_value=client
        ):
            main.store_vectors("test-document", {"name": "Test Candidate"})

        collection.upsert.assert_called_once()
        self.assertEqual(collection.upsert.call_args.kwargs["ids"], ["test-document"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
