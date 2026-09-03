## Backend service

The top-level project exposes a FastAPI service that accepts a PDF, extracts
its text, sends the text through the NVIDIA LLM resume parser, and returns
validated structured data.

```bash
export NVIDIA_API_KEY=nvapi-your-key-here
uv run uvicorn main:app --reload
```

The same service can run in Docker. The image reads the scripts and root
configuration from `/app`; pass the local `.env` at runtime:

```bash
docker build -t myjobbuddy .
docker run --rm --env-file .env -p 8000:8000 -v "$(pwd)/chroma_db:/app/chroma_db" myjobbuddy
```

Change shared service settings in `config.py`. Change PDF utility paths in
`pdf2txt_config.py`, and resume batch-parser paths in
`resume_parser_config.py`. The child folders contain reusable functions; the
root `main.py` only assembles them into the FastAPI pipeline.

Upload a PDF with `POST /parse`:

```bash
curl -X POST http://localhost:8000/parse \
	-F "file=@resume.pdf"
```

Vector indexing is disabled by default. Set `parsedata2vectors = True` in
`config.py`, or enable it for one request with `?index=true`. Disable it for a
request with `?index=false`.

The referenced standalone projects remain available for their original
utilities:

1. UnwrapPDF [https://github.com/nayaksomkar/UnwrapPDF]
2. resumeDataParse [https://github.com/nayaksomkar/resumeDataParse]