# Mini Pipeline Tests

Each pipeline step has its own test file. The tests create a tiny one-page PDF
fixture and use fake LLM and Chroma clients, so they need no API request or
real external database write.

## Test Steps

### 1. PDF to text

File: `test/test_pdf_to_txt.py`

```python
text = extract_text(temporary_pdf_path)
assert "--- Page 1 ---" in text
```

Status: PASS

### 2. Text to LLM parse

File: `test/test_txt_to_llm.py`

```python
parsed_data = parse_text(extracted_text)
assert parsed_data["name"] == "Test Candidate"
```

Status: PASS using a fake chain. No API request was made.

### 3. Parsed data to external ChromaDB

File: `test/test_parse_to_chroma.py`

```python
store_vectors("test-document", {"name": "Test Candidate"})
collection.upsert.assert_called_once()
```

Status: PASS using fake embeddings and a fake external Chroma collection. No real database was changed.

### 4. Service status

File: `test/test_status.py`

```python
assert health() == {"status": "ok"}
```

Status: PASS

### 5. Complete pipeline

File: `test/test_pipeline.py`

```python
result = run_pipeline()
assert result["status"] == {"status": "ok"}
```

Status: PASS. The pipeline file only calls the four production functions in order.

### 6. HTTP health endpoint

File: `test/test_api.py`

```python
response = client.get("/health")
assert response.json() == {"status": "ok"}
```

Status: PASS using FastAPI's local test client. No external API request is made.

### 7. Prepared sample data endpoint

File: `test/test_api.py`

```python
response = client.get("/sample_data")
assert response.json()[0]["resume_id"] == 1
```

Status: PASS. Reads `data/resume.json`; no AI request is made.

### 8. Sample text endpoint

File: `test/test_api.py`

```python
response = client.get("/sample_resume_txt")
assert response.json()["files"]
```

Status: PASS. Reads the bundled files in `data/resume_txt/`.

## Run

```bash
uv sync --locked --group dev
uv run pytest -q
```

Expected result: 8 tests passed.
