"""HTTP-level tests for the FastAPI service."""

from fastapi.testclient import TestClient

import main
from main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sample_data_endpoint_does_not_need_ai():
    with TestClient(app) as client:
        response = client.get("/sample_data")

    assert response.status_code == 200
    assert response.json()[0]["resume_id"] == 1
    assert response.json()[0]["resume_content"]["name"] == "Aarav Sharma"


def test_single_sample_data_endpoint():
    with TestClient(app) as client:
        response = client.get("/sample_data/2")

    assert response.status_code == 200
    assert response.json()["resume_id"] == 2
    assert response.json()["resume_content"]["name"] == "Priya Nair"


def test_single_sample_data_endpoint_returns_not_found():
    with TestClient(app) as client:
        response = client.get("/sample_data/999")

    assert response.status_code == 404


def test_sample_resume_text_endpoint():
    with TestClient(app) as client:
        response = client.get("/sample_resume_txt")

    assert response.status_code == 200
    assert response.json()["files"]
    assert "Aarav" in response.json()["files"][0]["text"]


def test_single_sample_resume_text_endpoint_returns_raw_text():
    with TestClient(app) as client:
        response = client.get("/sample_resume_txt/resumeONE.txt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text.startswith("Aarav Sharma")


def test_single_sample_resume_text_endpoint_returns_not_found():
    with TestClient(app) as client:
        response = client.get("/sample_resume_txt/missing.txt")

    assert response.status_code == 404


def test_parse_accepts_txt_upload(monkeypatch):
    monkeypatch.setattr(
        main,
        "parse_text",
        lambda text: {"resume_content": {"name": text.splitlines()[0]}},
    )

    with TestClient(app) as client:
        response = client.post(
            "/parse",
            files={"file": ("resume.txt", b"Aarav Sharma\nPython", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["data"]["resume_content"]["name"] == "Aarav Sharma"