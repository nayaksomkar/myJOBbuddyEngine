"""Shared fixtures and fakes used by the individual pipeline tests."""

import fitz
from resumeDataParse.parserMain import Resume


def sample_pdf_bytes() -> bytes:
    """Return a tiny PDF fixture for the extraction stage."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Test Candidate\nPython")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


class FakeChain:
    """Return deterministic parsed data instead of calling the external LLM."""

    def invoke(self, values):
        self.received_text = values["resume_text"]
        return Resume(name="Test Candidate", skills=["Python"])


class FakeParser:
    """Provide the parser instructions expected by the LLM stage."""

    def get_format_instructions(self):
        return "return Resume JSON"
