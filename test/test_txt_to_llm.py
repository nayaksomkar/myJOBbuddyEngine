"""Step 2: test extracted text to structured LLM data."""

import unittest
from unittest.mock import patch

from resumeDataParse import main as resume_parser
from resumeDataParse.main import parse_text
from helpers import FakeChain, FakeParser


class TxtToLlmTests(unittest.TestCase):
    """Check that text becomes the expected structured data."""

    def test_txt_to_llm_parse(self):
        with patch.object(
            resume_parser, "build_chain", return_value=(FakeChain(), FakeParser())
        ):
            result = parse_text("--- Page 1 ---\nTest Candidate\nPython")
        self.assertEqual(result["name"], "Test Candidate")
        self.assertEqual(result["skills"], ["Python"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
