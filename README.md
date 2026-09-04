# myJOBbuddy Engine

myJOBbuddy Engine is a FastAPI service that extracts text from uploaded PDF
resumes, sends the text to an NVIDIA-hosted LangChain model, and returns a
validated structured resume. Optional ChromaDB indexing supports later
retrieval workflows.

## Architecture

```text
Client
  |
  v
FastAPI backend (one container)
  |
  +-- UnwrapPDF       PDF text extraction
  +-- resumeDataParse LLM prompt and Pydantic validation
  +-- pdf2txt         Optional embedding/indexing utility
  +-- ChromaDB        Optional local vector persistence
  |
  +-- NVIDIA API      External resume parsing model
```

The API entry point is `main.py`. The existing utility modules remain
separate and can still be used independently.

## Configuration

Copy `.env.example` to `.env` for local development and provide an NVIDIA API
key. Never commit `.env` or real credentials.

| Variable | Default | Purpose |
| --- | --- | --- |
| `NVIDIA_API_KEY` | required for `/parse` | NVIDIA model authentication |
| `LOG_LEVEL` | `INFO` | Python log level |
| `PORT` | `8000` | Application port used by direct startup |
| `MAX_UPLOAD_SIZE_BYTES` | `10485760` | Maximum PDF upload size |
| `PARSE_DATA_TO_VECTORS` | `false` | Index parsed data by default |
| `CHROMA_DB_PATH` | `./chroma_db` | Chroma persistence directory |

## Run Locally

Prerequisites: Python 3.11+, `uv`, and an NVIDIA API key for live parsing.

```bash
cp .env.example .env
# edit .env and set NVIDIA_API_KEY
uv sync --group dev
uv run uvicorn main:app --reload
```

The API listens on `http://localhost:8000`.

## Docker

Build and run the single application container:

```bash
docker build -t myjobbuddy:local .
docker run --rm --name myjobbuddy \
  --env-file .env \
  -p 8000:8000 \
  -v myjobbuddy-chroma:/app/chroma_db \
  myjobbuddy:local
```

The image runs as a non-root user and exposes only port `8000`. The Chroma
volume is optional when vector indexing is disabled. Mount it when indexed
data must survive container replacement. The NVIDIA model and its API remain
external services; the image does not contain model weights or secrets.

## API and Health Check

Check liveness without contacting NVIDIA or Chroma:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Parse a resume:

```bash
curl -X POST http://localhost:8000/parse -F "file=@resume.pdf"
curl -X POST "http://localhost:8000/parse?index=true" -F "file=@resume.pdf"
```

The service rejects non-PDF uploads, empty/unreadable PDFs, and uploads larger
than `MAX_UPLOAD_SIZE_BYTES`. Provider and indexing failures are logged with
tracebacks but return generic client-safe error messages.

## Tests

Tests use deterministic PDF, LLM, and Chroma fakes; they do not call NVIDIA or
write to the repository's Chroma database.

```bash
uv sync --group dev
uv run pytest -q
```

The GitHub Actions workflow runs the tests and builds the Docker image on
pushes and pull requests.

## Logging and Shutdown

Operational logs are written to stdout/stderr for `docker logs`:

```bash
docker logs myjobbuddy
docker stop myjobbuddy
```

Startup, shutdown, request status/duration, and external-operation failures
are logged. Credentials, request bodies, and provider exception details are
not written to responses or normal request logs. `docker stop` sends the
normal termination signal to Uvicorn for graceful shutdown.

## Deployment Notes

Build this image in CI and deploy it to a container host that can inject
`NVIDIA_API_KEY` as a secret and expose port `8000`. Configure a persistent
volume only if Chroma indexing is enabled. The container filesystem is
disposable; source PDFs and generated files should be supplied through an
external storage workflow if that becomes a product requirement.
