"""HTTP-level tests for the FastAPI service."""

from fastapi.testclient import TestClient

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


def test_sample_resume_text_endpoint():
    with TestClient(app) as client:
        response = client.get("/sample_resume_txt")

    assert response.status_code == 200
    assert response.json()["files"]
    assert "Aarav" in response.json()["files"][0]["text"]