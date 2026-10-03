"""Unit tests for CloudinaryStorage and migration manager with mocked API calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.storage.cloudinary_storage import CloudinaryStorage
from src.storage.migration import CloudinaryMigrationManager


@pytest.fixture
def mock_storage():
    """Create CloudinaryStorage with mocked SDK components."""
    storage = CloudinaryStorage(
        cloud_name="demo-cloud",
        api_key="1234567890",
        api_secret="super_secret_key_xyz",
    )
    mock_uploader = MagicMock()
    mock_api = MagicMock()
    mock_utils = MagicMock()

    storage._uploader = mock_uploader
    storage._api = mock_api
    storage._utils = mock_utils
    storage._is_configured = True
    return storage


def test_cloudinary_storage_config_and_secrets(mock_storage):
    """Test configuration initialization and secret masking."""
    assert mock_storage.is_configured is True
    assert mock_storage.cloud_name == "demo-cloud"
    assert "super_secret_key_xyz" not in repr(mock_storage)


def test_public_id_generation():
    """Test predictable user-scoped public_id path generation."""
    storage = CloudinaryStorage(cloud_name="demo", api_key="123", api_secret="sec")
    
    # Standard format
    pid = storage.build_public_id("usr_user123", "doc_paper456")
    assert pid == "researchmate/users/usr_user123/papers/doc_paper456.pdf"

    # Sanitization: path traversal must be stripped
    pid_dirty = storage.build_public_id("usr/../hack", "doc#$123")
    assert ".." not in pid_dirty
    assert pid_dirty.startswith("researchmate/users/usr___hack/papers/")


def test_mocked_pdf_upload_success(mock_storage):
    """Test successful PDF upload returning expected metadata."""
    mock_storage._uploader.upload.return_value = {
        "public_id": "researchmate/users/u1/papers/d1.pdf",
        "secure_url": "https://res.cloudinary.com/demo/raw/upload/researchmate/users/u1/papers/d1.pdf",
        "url": "http://res.cloudinary.com/demo/raw/upload/researchmate/users/u1/papers/d1.pdf",
        "bytes": 54321,
        "format": "pdf",
        "resource_type": "raw",
        "created_at": "2026-10-03T10:00:00Z",
    }

    result = mock_storage.upload_pdf(
        file_input=b"%PDF-1.4 Mock PDF Content",
        user_id="u1",
        document_id="d1",
        filename="test.pdf",
    )

    assert result["success"] is True
    assert result["public_id"] == "researchmate/users/u1/papers/d1.pdf"
    assert result["bytes"] == 54321
    assert result["error"] is None


def test_mocked_pdf_upload_failure(mock_storage):
    """Test graceful handling of upload exceptions without leaking secrets."""
    mock_storage._uploader.upload.side_effect = Exception("Cloudinary error with secret super_secret_key_xyz")

    result = mock_storage.upload_pdf(
        file_input=b"content",
        user_id="u1",
        document_id="d1",
    )

    assert result["success"] is False
    assert "super_secret_key_xyz" not in result["error"]
    assert "[REDACTED]" in result["error"]


def test_mocked_pdf_deletion(mock_storage):
    """Test asset deletion on Cloudinary."""
    mock_storage._uploader.destroy.return_value = {"result": "ok"}

    deleted = mock_storage.delete_pdf("u1", "d1")
    assert deleted is True


def test_mocked_asset_exists(mock_storage):
    """Test checking asset existence."""
    mock_storage._api.resource.return_value = {"public_id": "researchmate/users/u1/papers/d1.pdf"}
    assert mock_storage.asset_exists("u1", "d1") is True

    mock_storage._api.resource.side_effect = Exception("Not found")
    assert mock_storage.asset_exists("u1", "d2") is False


def test_migration_manager(tmp_path, mock_storage):
    """Test migrating local papers to Cloudinary without deleting by default."""
    mock_storage._uploader.upload.return_value = {
        "public_id": "researchmate/users/mig_user_1/papers/paper1.pdf",
        "secure_url": "https://cloudinary.com/test.pdf",
        "bytes": 1000,
    }

    # Setup fake local user folder
    user_id = "mig_user_1"
    user_papers_dir = tmp_path / "users" / user_id / "papers"
    user_papers_dir.mkdir(parents=True)
    fake_pdf = user_papers_dir / "paper1.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy paper content")

    with patch("src.storage.migration.get_data_dir", return_value=tmp_path):
        manager = CloudinaryMigrationManager(cloudinary_storage=mock_storage)
        res = manager.migrate_user_papers(user_id=user_id, delete_local=False)

        assert res["total_found"] == 1
        assert res["migrated"] == 1
        assert fake_pdf.exists()  # Local copy preserved by default
