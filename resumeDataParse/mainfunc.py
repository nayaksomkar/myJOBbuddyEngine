"""I/O helpers: reading resume files and persisting parsed results.

Kept separate from `main.py` so the orchestration logic stays readable and
these helpers can be unit-tested in isolation.
"""

import json
import os
from typing import List


def fetch_file_contents(folder_path: str) -> List[str]:
    """Read and return the text contents of every `.txt` file in a folder.

    Non-text files and hidden files (e.g. `.DS_Store`) are skipped, and the
    files are processed in sorted order so the output `resume.json` is
    deterministic across runs.

    Args:
        folder_path: Directory containing the plain-text resumes.

    Returns:
        List of file contents, one entry per `.txt` file.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Resume folder not found: {folder_path!r}. "
            "Place your .txt resumes in this folder or update config.folder_path."
        )

    # Keep only visible plain-text files so unrelated data is ignored.
    file_contents = []

    # Sort so resumeONE.txt, resumeTWO.txt, ... are always parsed in order.
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)

        # Only read plain text files; skip directories and hidden files.
        if not file_name.endswith(".txt") or file_name.startswith("."):
            continue

        with open(file_path, "r", encoding="utf-8") as file:
            file_contents.append(file.read())

    return file_contents


def append_resume_json(file_name: str, resume_content: dict, resume_id: int) -> None:
    """Append a structured resume to a JSON array file.

    Creates the file with an empty array if it does not exist yet or is empty,
    then appends a new entry of the form:
        {"resume_id": <id>, "resume_content": <parsed fields>}

    Args:
        file_name: Path to the output JSON file.
        resume_content: Parsed resume fields as a dict.
        resume_id: Sequential ID assigned to this resume.
    """
    # Read the existing array before appending the next parsed resume.
    data: list = []

    # Load any previously parsed entries so we append instead of overwriting.
    if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
        with open(file_name, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

    data.append({"resume_id": resume_id, "resume_content": resume_content})

    # ensure_ascii=False keeps non-ASCII characters (e.g. "–") human-readable.
    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
