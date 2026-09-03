"""FastAPI service for PDF extraction, resume parsing, and vector indexing."""

import hashlib
import json
import tempfile
from pathlib import Path

import chromadb
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from config import chroma_collection, chroma_db_path, parsedata2vectors
from UnwrapPDF.pdf_extract import extract_text
from resumeDataParse.main import parse_text

# Load the single repository-level environment file before creating the LLM.
load_dotenv(Path(__file__).resolve().parent / ".env")


app = FastAPI(title="myJOBbuddy Engine")


def store_vectors(document_id: str, parsed_data: dict) -> None:
    """Embed one parsed document and persist it in the configured collection."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_or_create_collection(name=chroma_collection)
    document = json.dumps(parsed_data, ensure_ascii=False)
    embedding = model.encode([document]).tolist()
    collection.upsert(
        ids=[document_id],
        embeddings=embedding,
        documents=[document],
        metadatas=[{"document_id": document_id}],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/parse")
async def parse_pdf(
    file: UploadFile = File(...),
    index: bool | None = Query(default=None),
) -> dict:
    """Extract and parse one uploaded PDF, optionally indexing its result."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    suffix = Path(file.filename or "document.pdf").suffix or ".pdf"
    # A temporary file lets the shared extractor process the upload by path.
    pdf_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary_file:
        temporary_file.write(pdf_bytes)
        temporary_file.flush()
        extracted_text = extract_text(temporary_file.name)

    if not extracted_text.strip():
        raise HTTPException(status_code=422, detail="The PDF contains no extractable text")

    try:
        parsed_data = parse_text(extracted_text)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"LLM parsing failed: {error}") from error

    should_index = parsedata2vectors if index is None else index
    if should_index:
        document_id = hashlib.sha256(pdf_bytes).hexdigest()
        try:
            store_vectors(document_id, parsed_data)
        except Exception as error:
            raise HTTPException(status_code=502, detail=f"Vector indexing failed: {error}") from error

    return {"filename": file.filename, "indexed": should_index, "data": parsed_data}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
