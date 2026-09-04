

"""Shared runtime settings for the FastAPI service and parser functions."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
	"""Parse common environment boolean spellings."""
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


parsedata2vectors = _as_bool(os.getenv("PARSE_DATA_TO_VECTORS"))

llm_model = "openai/gpt-oss-20b"
llm_temperature = 0.8
llm_top_p = 1
chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
chroma_collection = "parsed_documents"
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
port = int(os.getenv("PORT", "8000"))
max_upload_size_bytes = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))

prompt = """
You are an expert resume information extraction system.

Extract only information present in the provided resume text. Do not invent
missing details. Return empty strings or empty lists for unavailable fields.
Extract skills, work experience, projects, education, and certifications as
separate entries, and create a concise professional summary.

{format_instructions}

Resume Text:

{resume_text}
"""

