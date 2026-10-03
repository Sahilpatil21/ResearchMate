"""Tests for MongoDB/Local Authentication and Multi-Tenant User Isolation."""

import pytest
from src.auth.auth_service import AuthService
from src.auth.user_model import User
from src.utils.file_utils import clear_all_documents, get_user_data_dir, load_all_processed_documents


def test_user_model_serialization():
    """Test User model dictionary creation and deserialization."""
    user = User(email="researcher@university.edu", full_name="Dr. Jane Doe", password_hash="hashed_secret")
    assert user.user_id.startswith("usr_")
    assert user.email == "researcher@university.edu"
    assert user.full_name == "Dr. Jane Doe"

    d = user.to_dict(include_hash=False)
    assert "password_hash" not in d
    assert d["email"] == "researcher@university.edu"

    d_with_hash = user.to_dict(include_hash=True)
    assert d_with_hash["password_hash"] == "hashed_secret"

    restored = User.from_dict(d_with_hash)
    assert restored.email == user.email
    assert restored.user_id == user.user_id


def test_auth_service_password_hashing():
    """Test bcrypt password hashing and verification."""
    auth = AuthService()
    pwd = "secure_password_123"
    hashed = auth.hash_password(pwd)

    assert hashed != pwd
    assert auth.verify_password(pwd, hashed) is True
    assert auth.verify_password("wrong_password", hashed) is False


def test_auth_service_registration_and_authentication(tmp_path):
    """Test registering a new user and authenticating."""
    auth = AuthService()
    # Use temporary file for test isolation
    auth._local_auth_path = tmp_path / "_test_users.json"
    auth.is_mongo_connected = False
    auth._users_collection = None

    # Test invalid email
    ok, msg, user = auth.register_user("invalid_email", "password123", "Test User")
    assert ok is False
    assert "valid email" in msg

    # Test short password
    ok, msg, user = auth.register_user("test@example.com", "123", "Test User")
    assert ok is False
    assert "6 characters" in msg

    # Test valid registration
    ok, msg, user = auth.register_user("test@example.com", "password123", "Dr. Test")
    assert ok is True
    assert user is not None
    assert user.email == "test@example.com"
    assert user.full_name == "Dr. Test"

    # Test duplicate registration
    ok_dup, msg_dup, _ = auth.register_user("test@example.com", "password123", "Dr. Test")
    assert ok_dup is False
    assert "already exists" in msg_dup

    # Test successful login
    auth_ok, auth_msg, auth_user = auth.authenticate_user("test@example.com", "password123")
    assert auth_ok is True
    assert auth_user is not None
    assert auth_user.user_id == user.user_id

    # Test failed login with wrong password
    bad_ok, bad_msg, _ = auth.authenticate_user("test@example.com", "wrong_pass")
    assert bad_ok is False


def test_multi_tenant_user_storage_paths():
    """Test user directory isolation in file_utils."""
    user_a_dir = get_user_data_dir("usr_alice123")
    user_b_dir = get_user_data_dir("usr_bob456")

    assert "usr_alice123" in str(user_a_dir)
    assert "usr_bob456" in str(user_b_dir)
    assert user_a_dir != user_b_dir
