from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import Settings, get_settings
from backend.main import app

MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n"
    b"<< /Type /Catalog >>\n"
    b"endobj\n"
    b"trailer\n"
    b"<< /Root 1 0 R >>\n"
    b"%%EOF"
)


def create_test_settings(tmp_path: Path) -> Settings:

    return Settings(
        app_env="test",
        data_directory=tmp_path,
        regulation_directory=tmp_path / "regulations",
        policy_directory=tmp_path / "policies",
        processed_directory=tmp_path / "processed",
        faiss_directory=tmp_path / "faiss",
        max_upload_size_mb=1,
    )


def test_health_endpoint(tmp_path: Path):

    settings = create_test_settings(tmp_path)

    app.dependency_overrides[get_settings] = lambda: settings

    try:
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    finally:
        app.dependency_overrides.clear()


def test_upload_policy_pdf(tmp_path: Path):

    settings = create_test_settings(tmp_path)

    app.dependency_overrides[get_settings] = lambda: settings

    try:
        with TestClient(app) as client:
            response = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "Records Policy.pdf",
                        MINIMAL_PDF,
                        "application/pdf",
                    )
                },
                data={
                    "document_type": "policy",
                    "title": "Records Management Policy",
                },
            )

        assert response.status_code == 201

        payload = response.json()

        assert payload["document"]["document_type"] == "policy"
        assert payload["document"]["original_filename"] == "Records Policy.pdf"

    finally:
        app.dependency_overrides.clear()


def test_reject_invalid_upload(tmp_path: Path):

    settings = create_test_settings(tmp_path)

    app.dependency_overrides[get_settings] = lambda: settings

    try:
        with TestClient(app) as client:
            response = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "notes.txt",
                        b"Not a PDF",
                        "text/plain",
                    )
                },
                data={
                    "document_type": "policy",
                },
            )

        assert response.status_code == 415

    finally:
        app.dependency_overrides.clear()
