# myJOBbuddy Engine

A FastAPI service that reads PDF resumes and returns structured resume data.
New PDFs use NVIDIA AI. Prepared sample data can be read without AI.

## How It Works

```text
Client -> FastAPI Docker container -> PDF extraction -> NVIDIA AI -> Pydantic validation
                                                        |
                                                        +-> Optional external ChromaDB
```

The project runs the API in one container. ChromaDB is a separate external
service; this container does not host or store the Chroma database.

## Main Folders

| Path | Purpose |
| --- | --- |
| `main.py` | FastAPI routes and application startup |
| `UnwrapPDF/` | PDF extraction and external Chroma utility scripts |
| `resumeDataParse/` | AI parsing and the Pydantic resume schema |
| `data/resume.json` | Prepared parsed sample resumes |
| `data/resume_txt/` | Sample resume text files |
| `config/` | Application settings and environment files |
| `docker/` | Dockerfile and container entrypoint |
| `test/` | Automated tests |

## Pydantic

Pydantic checks that parsed resumes have the expected fields: name, email,
phone, skills, experience, projects, education, certifications, and summary.
It works like a form checker between the AI response and the API response.

## What Is Inside the Container?

The image contains:

- Python 3.11 and locked packages from `pyproject.toml` and `uv.lock`.
- FastAPI, Uvicorn, Pydantic, PyMuPDF, LangChain, ChromaDB client, and the
  NVIDIA embeddings client.
- Application code under `/app`.
- Sample data under `/app/data/`.
- `/app/temp/` for short-lived upload files.
- `/app/logs/` for future file-based logs.
- `/entrypoint.sh`, which starts Uvicorn using `PORT` or `8000`.

The container runs as non-root user `appuser`, exposes port `8000`, and has a
health check for `/health`. Normal logs are written to Docker stdout.

The image does not contain `config/.env`, API keys, tests, Git files, or any
ChromaDB database files. ChromaDB is accessed over HTTP using environment
settings.

## Lightweight Embeddings

The API uses NVIDIA's hosted embeddings service when Chroma indexing is
requested. This keeps embedding work outside the container and avoids local
`sentence-transformers`, PyTorch, Triton, and CUDA packages. Set
`NVIDIA_EMBEDDING_MODEL` to choose the NVIDIA embedding model.

## Configuration

Create the local environment file:

```bash
cp config/.env.example config/.env
```

Set `NVIDIA_API_KEY` in `config/.env`. Never commit that file.

| Variable | Purpose | Default |
| --- | --- | --- |
| `NVIDIA_API_KEY` | Enables new PDF parsing | Required for `/parse` |
| `CHROMA_HOST` | External ChromaDB hostname | `localhost` |
| `CHROMA_PORT` | External ChromaDB port | `8001` |
| `CHROMA_SSL` | Use HTTPS for ChromaDB | `false` |
| `CHROMA_COLLECTION` | ChromaDB collection name | `parsed_documents` |
| `PORT` | API port | `8000` |
| `LOG_LEVEL` | Log detail level | `INFO` |
| `MAX_UPLOAD_SIZE_BYTES` | Maximum PDF size | `10485760` |
| `TEMP_DATA_PATH` | Temporary file directory | `./temp` |
| `LOGS_PATH` | Optional file-log directory | `./logs` |

## API Routes

| Method | Route | Purpose | AI needed? |
| --- | --- | --- | --- |
| `GET` | `/health` | Check service status | No |
| `GET` | `/sample_data` | Return prepared parsed resumes | No |
| `GET` | `/sample_data/{resume_id}` | Return one prepared parsed resume | No |
| `GET` | `/sample_resume_txt` | Return sample text files | No |
| `GET` | `/sample_resume_txt/{filename}` | Return one raw sample resume text file | No |
| `POST` | `/parse` | Parse an uploaded PDF | Yes |
| `POST` | `/parse?index=true` | Parse and send data to external ChromaDB | Yes |

## Test Locally

Requirements: Python 3.11+ and `uv`.

```bash
uv sync --locked --group dev
uv run pytest -q
```

Expected result: `8 passed`.

Start the API:

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Test the no-AI routes from another terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/sample_data
curl http://localhost:8000/sample_resume_txt
```

Fetch one parsed resume or one raw resume text file:

```bash
curl http://localhost:8000/sample_data/1
curl http://localhost:8000/sample_resume_txt/resumeONE.txt
```

Stop the server with `Ctrl+C`.

## Build and Run Docker

Make sure Docker is running, then build from the project root:

```bash
docker build -f docker/Dockerfile -t myjobbuddy:local .
```

Run the API container. No Chroma volume is mounted because ChromaDB is
external:

```bash
docker run --rm --name myjobbuddy \
  --env-file config/.env \
  -p 8000:8000 \
  myjobbuddy:local
```

Test the container:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/sample_data
curl http://localhost:8000/sample_resume_txt
```

Parse a PDF or TXT resume:

```bash
curl -X POST http://localhost:8000/parse \
  -F "file=@resume.pdf"

curl -X POST http://localhost:8000/parse \
  -F "file=@resume.txt"

curl -X POST http://localhost:8000/parse \
  -F "file=@data/resume_txt/resumeONE.txt"
```

The parser first extracts all readable text, creates a detail-preserving
comprehensive summary, and then converts that summary into the structured
resume response. Parsing uses the configured NVIDIA model with deterministic
temperature `0`.

Index a parsed PDF in the external ChromaDB service:

```bash
curl -X POST "http://localhost:8000/parse?index=true" \
  -F "file=@resume.pdf"
```

View logs and stop the container:

```bash
docker logs myjobbuddy
docker stop myjobbuddy
```

Inspect temporary and optional log files:

```bash
docker exec myjobbuddy sh -c 'find /app/temp -maxdepth 1 -type f -print'
docker exec myjobbuddy sh -c 'find /app/logs -maxdepth 1 -type f -print'
```

Uploaded temporary files are removed after processing. Current logs use
`docker logs`; the `logs/` folder is reserved for future JSON, SQLite, or text
logs.

## Hosting

1. Run the tests and local Docker checks.
2. Build and push `myjobbuddy:local` to a container registry.
3. Configure the hosting provider to run the image.
4. Add `NVIDIA_API_KEY` as a secret.
5. Set the provider's `PORT`; the entrypoint uses it automatically.
6. Set the health check path to `/health`.
7. Allow outbound HTTPS access to NVIDIA.
8. Set `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SSL`, and
   `CHROMA_COLLECTION` for the external ChromaDB service.

The sample routes work without NVIDIA. The `/parse` route needs NVIDIA, and
`/parse?index=true` also needs the external ChromaDB service. No database,
Chroma volume, or extra database container is created by this project.
