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


llm_provider = "mistral"
llm_model = "open-mistral-nemo"
llm_temperature = 0
llm_top_p = 1
llm_max_tokens = 4096
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

summary_prompt = """
You are preparing a complete, factual representation of a resume for a second
structured extraction step.

Read every line of the source text. Produce a comprehensive summary that keeps
all facts and every distinct item, including contact details, skills,
competencies, responsibilities, achievements, projects, education,
certifications, languages, interests, links, dates, employers, technologies,
and headings. Preserve names, dates, numbers, and technical terms exactly when
possible. Do not invent or merge items. Organize the result under clear
headings, but do not omit details just to make it shorter.

Source resume text:

{resume_text}
"""

prompt = """
You are an expert resume information extraction system.

Extract all information present in the resume representation below. Do not
invent, drop, or merge details. Return empty strings or empty lists only when
the information is genuinely unavailable.

Put every distinct technical or professional competency in skills. Put each
employer/role and its responsibilities or achievements in a separate
experience entry. Put every named project and its description in a separate
project entry. Put every degree or course in education, and every certificate
or training item in certifications. Include dates and organizations in the
corresponding entries. Create a concise summary based only on the source.

{format_instructions}

Comprehensive resume representation:

{resume_text}
"""