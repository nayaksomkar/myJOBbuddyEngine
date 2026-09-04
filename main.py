"""FastAPI service for PDF extraction and resume parsing."""

import json
import hashlib
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
import chromadb
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from config.config import (
    chroma_collection,
    chroma_host,
    chroma_port,
    chroma_ssl,
    embedding_model,
    log_level,
    logs_path,
    max_upload_size_bytes,
    port,
    sample_data_path,
    sample_resume_text_path,
    temp_data_path,
)
from UnwrapPDF.pdf_extract import extract_text
from resumeDataParse.main import parse_text


logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("myjobbuddy")


@asynccontextmanager
async def lifespan(_: FastAPI):
    Path(temp_data_path).mkdir(parents=True, exist_ok=True)
    Path(logs_path).mkdir(parents=True, exist_ok=True)
    logger.info("service_started port=%s", port)
    yield
    logger.info("service_stopped")


app = FastAPI(title="myJOBbuddy Engine", lifespan=lifespan)


def store_vectors(document_id: str, parsed_data: dict) -> None:
    """Embed parsed data and send it to the external ChromaDB service."""
    model = NVIDIAEmbeddings(model=embedding_model)
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port, ssl=chroma_ssl)
    collection = client.get_or_create_collection(name=chroma_collection)
    document = json.dumps(parsed_data, ensure_ascii=False)
    embedding = model.embed_documents([document])
    collection.upsert(
        ids=[document_id],
        embeddings=embedding,
        documents=[document],
        metadatas=[{"document_id": document_id}],
    )


@app.middleware("http")
async def request_logging(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sample_data")
def get_sample_data() -> list[dict]:
    """Return prepared resume results without calling the AI service."""
    try:
        with sample_data_path.open("r", encoding="utf-8") as data_file:
            return json.load(data_file)
    except (OSError, json.JSONDecodeError) as error:
        logger.exception("sample_data_read_failed path=%s", sample_data_path)
        raise HTTPException(status_code=500, detail="Sample data is unavailable") from error


@app.get("/sample_data/{resume_id}")
def get_sample_resume(resume_id: int) -> dict:
    """Return one prepared resume result without calling the AI service."""
    try:
        with sample_data_path.open("r", encoding="utf-8") as data_file:
            sample_data = json.load(data_file)
    except (OSError, json.JSONDecodeError) as error:
        logger.exception("sample_data_read_failed path=%s", sample_data_path)
        raise HTTPException(status_code=500, detail="Sample data is unavailable") from error

    resume = next(
        (item for item in sample_data if item.get("resume_id") == resume_id),
        None,
    )
    if resume is None:
        raise HTTPException(status_code=404, detail="Sample resume not found")
    return resume


@app.get("/sample_resume_txt")
def get_sample_resume_text() -> dict[str, list[dict[str, str]]]:
    """Return the bundled sample resume text files without AI processing."""
    try:
        files = []
        for file_path in sorted(sample_resume_text_path.glob("*.txt")):
            files.append(
                {"filename": file_path.name, "text": file_path.read_text(encoding="utf-8")}
            )
        return {"files": files}
    except OSError as error:
        logger.exception("sample_text_read_failed path=%s", sample_resume_text_path)
        raise HTTPException(status_code=500, detail="Sample text is unavailable") from error


@app.get("/sample_resume_txt/{filename}", response_class=PlainTextResponse)
def get_sample_resume_text_file(filename: str) -> str:
    """Return one bundled sample resume as raw text."""
    file_path = (sample_resume_text_path / filename).resolve()
    if file_path.parent != sample_resume_text_path.resolve() or file_path.suffix != ".txt":
        raise HTTPException(status_code=404, detail="Sample resume text not found")

    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Sample resume text not found") from error
    except OSError as error:
        logger.exception("sample_text_read_failed path=%s", file_path)
        raise HTTPException(status_code=500, detail="Sample text is unavailable") from error


@app.post("/parse")
async def parse_pdf(
    file: UploadFile = File(...),
    index: bool = False,
) -> dict:
    """Extract and parse one uploaded PDF."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    suffix = Path(file.filename or "document.pdf").suffix or ".pdf"
    # A temporary file lets the shared extractor process the upload by path.
    pdf_bytes = await file.read(max_upload_size_bytes + 1)
    if len(pdf_bytes) > max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Uploaded PDF is too large")

    with tempfile.NamedTemporaryFile(suffix=suffix, dir=temp_data_path) as temporary_file:
        temporary_file.write(pdf_bytes)
        temporary_file.flush()
        try:
            extracted_text = extract_text(temporary_file.name)
        except Exception as error:
            logger.exception("pdf_extraction_failed filename=%s", file.filename)
            raise HTTPException(status_code=422, detail="The PDF could not be read") from error

    if not extracted_text.strip():
        raise HTTPException(status_code=422, detail="The PDF contains no extractable text")

    try:
        parsed_data = parse_text(extracted_text)
    except Exception as error:
        logger.exception("llm_parsing_failed filename=%s", file.filename)
        raise HTTPException(status_code=502, detail="Resume parsing service failed") from error

    if index:
        document_id = hashlib.sha256(pdf_bytes).hexdigest()
        try:
            store_vectors(document_id, parsed_data)
        except Exception as error:
            logger.exception("vector_indexing_failed filename=%s", file.filename)
            raise HTTPException(status_code=502, detail="External vector service failed") from error

    return {"filename": file.filename, "indexed": index, "data": parsed_data}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
