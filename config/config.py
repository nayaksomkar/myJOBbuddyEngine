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


llm_model = "openai/gpt-oss-20b"
llm_temperature = 0.8
llm_top_p = 1
chroma_host = os.getenv("CHROMA_HOST", "localhost")
chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
chroma_ssl = _as_bool(os.getenv("CHROMA_SSL"), default=False)
chroma_collection = os.getenv("CHROMA_COLLECTION", "parsed_documents")
embedding_model = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
project_root = Path(__file__).resolve().parent.parent
sample_data_path = project_root / "data" / "resume.json"
sample_resume_text_path = project_root / "data" / "resume_txt"
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
port = int(os.getenv("PORT", "8000"))
max_upload_size_bytes = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))
temp_data_path = os.getenv("TEMP_DATA_PATH", "./temp")
logs_path = os.getenv("LOGS_PATH", "./logs")

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