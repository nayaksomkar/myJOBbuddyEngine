

"""Shared runtime settings for the FastAPI service and parser functions."""


parsedata2vectors = False

llm_model = "openai/gpt-oss-20b"
llm_temperature = 0.8
llm_top_p = 1
chroma_db_path = "./chroma_db"
chroma_collection = "parsed_documents"

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

