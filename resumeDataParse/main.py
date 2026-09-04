"""Standalone batch entry point for parsing shared resume text files.

Pipeline:
    1. Load the API key from `config/.env`.
    2. Build a LangChain chain: PromptTemplate -> ChatMistralAI (LLM) -> Pydantic parser.
    3. Read every `.txt` resume from `resume_txt/`.
    4. Run each resume through the chain and append the structured result to
    `data/resume.json`.

Run from the project root:
    uv run resume-parse
    # or
    python main.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from .mainfunc import append_resume_json, fetch_file_contents
from .parserMain import Resume
from config.config import llm_model, llm_temperature, llm_top_p, llm_max_tokens, summary_prompt
from config.resume_parser_config import folder_path, output_file, prompt

# This child directory owns the script, while shared data and secrets live in config/.
PROJECT_ROOT = Path(__file__).resolve().parent

# Load secrets from config/.env into os.environ.
load_dotenv(PROJECT_ROOT.parent / "config" / ".env")


def _build_llm():
    """Configure the hosted model used for both resume-processing stages."""
    return ChatMistralAI(
        model=llm_model,
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=llm_temperature,
        top_p=llm_top_p,
        max_tokens=llm_max_tokens,
    )


def build_summary_chain():
    """Build the detail-preserving resume summarization stage."""
    prompt_template = ChatPromptTemplate.from_template(summary_prompt)
    return prompt_template | _build_llm() | StrOutputParser()


def build_chain():
    """Assemble the LangChain extraction pipeline.

    Returns:
        A callable chain that takes a dict with `resume_text` and
        `format_instructions` keys and returns a validated `Resume` object.
    """
    llm = _build_llm()

    # Parser turns the LLM's free-form text output into a Resume object,
    # and provides the schema instructions injected into the prompt.
    parser = PydanticOutputParser(pydantic_object=Resume)

    prompt_template = ChatPromptTemplate.from_template(prompt)

    # LangChain "pipe" syntax: prompt -> LLM -> parser, all in one callable.
    # Compose prompt formatting, model invocation, and schema validation.
    chain = prompt_template | llm | parser
    return chain, parser


def parse_text(extracted_text: str) -> dict:
    """Summarize all extracted text, then parse it into resume fields."""
    summary_chain = build_summary_chain()
    chain, parser = build_chain()
    comprehensive_summary = summary_chain.invoke({"resume_text": extracted_text})
    parsed = chain.invoke(
        {
            "resume_text": comprehensive_summary,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    return parsed.model_dump()


def main() -> None:
    """Orchestrate the full parse-and-persist pipeline."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY is not set. "
                "Copy config/.env.example to config/.env and fill in your key "
            "(get one at https://console.mistral.ai/)."
        )

    # Build the chain once and reuse it for every input file.
    summary_chain = build_summary_chain()
    chain, parser = build_chain()

    # Folder path from config/resume_parser_config.py is relative to the project root.
    input_folder = str(PROJECT_ROOT.parent / folder_path)
    output_path = PROJECT_ROOT.parent / output_file

    # Read all shared inputs in deterministic filename order.
    resume_contents = fetch_file_contents(input_folder)

    for resume_id, resume_text in enumerate(resume_contents, start=1):
        print(f"Parsing resume {resume_id}/{len(resume_contents)} ...")

        comprehensive_summary = summary_chain.invoke({"resume_text": resume_text})
        result = chain.invoke(
            {
                "resume_text": comprehensive_summary,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # Persist the validated Pydantic object as a dict.
        append_resume_json(
            file_name=str(output_path),
            resume_content=result.model_dump(),
            resume_id=resume_id,
        )

    print(f"Done. Results written to {output_path}")


if __name__ == "__main__":
    main()
