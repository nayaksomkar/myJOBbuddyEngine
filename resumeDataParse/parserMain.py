"""Pydantic schema for structured resume data.

This model is used two ways:
1. As the output schema handed to the LLM via `PydanticOutputParser`, so the
   model knows exactly which fields to produce.
2. As the validation layer that guarantees the parsed result matches the
   expected structure before it is persisted to JSON.
"""

from pydantic import BaseModel, Field


class Resume(BaseModel):
    """Structured resume data extracted from raw text.

    Every field defaults to an empty value so a resume missing a section
    still parses successfully instead of raising a validation error.
    """

    # Contact fields remain empty when the source resume omits them.
    name: str = Field(default="", description="Full name of the candidate")
    email: str = Field(default="", description="Candidate email address")
    phone: str = Field(default="", description="Candidate phone number")
    skills: list[str] = Field(
        default_factory=list, description="Technical and professional skills"
    )
    experience: list[str] = Field(
        default_factory=list, description="Work experience summaries"
    )
    projects: list[str] = Field(
        default_factory=list, description="Project titles and summaries"
    )
    education: list[str] = Field(
        default_factory=list, description="Educational qualifications"
    )
    certifications: list[str] = Field(
        default_factory=list, description="Certifications and courses"
    )
    summary: str = Field(default="", description="Professional summary of the candidate")
