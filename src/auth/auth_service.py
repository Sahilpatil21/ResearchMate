"""Authentication service with MongoDB persistence and resilient local fallback.

Handles password hashing (bcrypt), user registration, authentication,
and session validation.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bcrypt
from dotenv import load_dotenv

from src.auth.user_model import User
from src.utils.file_utils import get_data_dir

load_dotenv()


class AuthService:
    """Authentication and user management service supporting MongoDB with local fallback."""

    _instance: Optional[AuthService] = None

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> None:
        """Initialize AuthService.

        Args:
            mongo_uri: MongoDB connection string (e.g. 'mongodb://localhost:27017').
            db_name: MongoDB database name (default: 'researchmate_db').
        """
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", "researchmate_db")
        self.is_mongo_connected = False
        self._mongo_client = None
        self._users_collection = None
        
        # Local JSON store fallback path: data/users/_users_auth.json
        self._local_auth_path = get_data_dir() / "users" / "_users_auth.json"
        self._local_auth_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_connection()
        self._ensure_default_user()

    def _ensure_default_user(self) -> None:
        """Seed default local researcher account if not present."""
        try:
            demo_email = "researcher@local"
            user = self.get_user_by_email(demo_email)
            if not user:
                self.register_user(
                    email=demo_email,
                    password="password123",
                    full_name="Lead Researcher",
                )
        except Exception:
            pass

    def _init_connection(self) -> None:
        """Attempt to connect to MongoDB with a short timeout."""
        try:
            import pymongo
            client = pymongo.MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000,
            )
            # Trigger server connection check
            client.admin.command("ping")
            self._mongo_client = client
            db = self._mongo_client[self.db_name]
            self._users_collection = db["users"]
            self._sessions_collection = db["sessions"]
            # Ensure unique index on email and session token
            self._users_collection.create_index("email", unique=True)
            self._sessions_collection.create_index("token", unique=True)
            self.is_mongo_connected = True
        except Exception:
            self.is_mongo_connected = False
            self._mongo_client = None
            self._users_collection = None
            self._sessions_collection = None

    @classmethod
    def get_instance(cls) -> AuthService:
        """Singleton accessor for AuthService."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def hash_password(plain_password: str) -> str:
        """Hash plain text password using bcrypt."""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify plain text password against stored bcrypt hash."""
        try:
            return bcrypt.checkpw(
                plain_password.encode("utf-8"),
                hashed_password.encode("utf-8"),
            )
        except Exception:
            return False

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format with regex."""
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return bool(re.match(pattern, email.strip()))

    def _load_local_users(self) -> Dict[str, Dict[str, Any]]:
        """Load local fallback user records."""
        if not self._local_auth_path.exists():
            return {}
        try:
            content = self._local_auth_path.read_text(encoding="utf-8")
            return json.loads(content)
        except Exception:
            return {}

    def _save_local_users(self, users_dict: Dict[str, Dict[str, Any]]) -> None:
        """Save local fallback user records."""
        self._local_auth_path.parent.mkdir(parents=True, exist_ok=True)
        self._local_auth_path.write_text(
            json.dumps(users_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def register_user(
        self,
        email: str,
        password: str,
        full_name: str,
    ) -> Tuple[bool, str, Optional[User]]:
        """Register a new user account.

        Args:
            email: User's email address.
            password: User's chosen password (min 6 characters).
            full_name: User's full name.

        Returns:
            Tuple of (success_boolean, message, user_object_or_none).
        """
        clean_email = email.lower().strip()
        clean_name = full_name.strip()

        if not clean_email or not self.validate_email(clean_email):
            return False, "Please enter a valid email address.", None

        if len(password) < 6:
            return False, "Password must be at least 6 characters long.", None

        if len(clean_name) < 2:
            return False, "Please enter your full name (at least 2 characters).", None

        # Check for existing email in MongoDB or local store
        existing_user = self.get_user_by_email(clean_email)
        if existing_user:
            return False, f"An account with email '{clean_email}' already exists. Please log in.", None

        pwd_hash = self.hash_password(password)
        new_user = User(
            email=clean_email,
            full_name=clean_name,
            password_hash=pwd_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Save to MongoDB or local fallback
        if self.is_mongo_connected and self._users_collection is not None:
            try:
                doc = new_user.to_dict(include_hash=True)
                self._users_collection.insert_one(doc)
                return True, "Registration successful!", new_user
            except Exception as e:
                # If unique constraint violation or error, fallback to local save
                pass

        # Local fallback save
        local_users = self._load_local_users()
        local_users[new_user.user_id] = new_user.to_dict(include_hash=True)
        self._save_local_users(local_users)
        return True, "Registration successful!", new_user

    def authenticate_user(
        self,
        email: str,
        password: str,
    ) -> Tuple[bool, str, Optional[User]]:
        """Authenticate user credentials.

        Args:
            email: Registered user email.
            password: User password.

        Returns:
            Tuple of (success_boolean, message, user_object_or_none).
        """
        clean_email = email.lower().strip()
        if not clean_email or not password:
            return False, "Please provide both email and password.", None

        user = self.get_user_by_email(clean_email)
        if not user:
            return False, "Invalid email or password.", None

        if not self.verify_password(password, user.password_hash):
            return False, "Invalid email or password.", None

        # Update last login timestamp
        now_iso = datetime.now(timezone.utc).isoformat()
        user.last_login = now_iso

        if self.is_mongo_connected and self._users_collection is not None:
            try:
                self._users_collection.update_one(
                    {"user_id": user.user_id},
                    {"$set": {"last_login": now_iso}},
                )
            except Exception:
                pass
        else:
            local_users = self._load_local_users()
            if user.user_id in local_users:
                local_users[user.user_id]["last_login"] = now_iso
                self._save_local_users(local_users)

        return True, "Login successful!", user

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Fetch user by email."""
        clean_email = email.lower().strip()
        if self.is_mongo_connected and self._users_collection is not None:
            try:
                doc = self._users_collection.find_one({"email": clean_email})
                if doc:
                    return User.from_dict(doc)
            except Exception:
                pass

        local_users = self._load_local_users()
        for u_data in local_users.values():
            if u_data.get("email", "").lower() == clean_email:
                return User.from_dict(u_data)

        return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Fetch user by user_id."""
        if self.is_mongo_connected and self._users_collection is not None:
            try:
                doc = self._users_collection.find_one({"user_id": user_id})
                if doc:
                    return User.from_dict(doc)
            except Exception:
                pass

        local_users = self._load_local_users()
        if user_id in local_users:
            return User.from_dict(local_users[user_id])
        return None

    def get_all_users_count(self) -> int:
        """Return total registered user count."""
        if self.is_mongo_connected and self._users_collection is not None:
            try:
                return self._users_collection.count_documents({})
            except Exception:
                pass
        return len(self._load_local_users())

    def create_session(self, user_id: str) -> str:
        """Create and store a persistent session token for the user."""
        import secrets
        token = secrets.token_urlsafe(32)
        now_iso = datetime.now(timezone.utc).isoformat()
        if self.is_mongo_connected and self._sessions_collection is not None:
            try:
                self._sessions_collection.insert_one({
                    "token": token,
                    "user_id": user_id,
                    "created_at": now_iso,
                })
                return token
            except Exception:
                pass
        
        # Local fallback session file
        session_file = self._local_auth_path.parent / "_sessions.json"
        sessions = {}
        if session_file.exists():
            try:
                sessions = json.loads(session_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        sessions[token] = {"user_id": user_id, "created_at": now_iso}
        session_file.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
        return token

    def get_user_by_session_token(self, token: str) -> Optional[User]:
        """Validate a session token and return the associated User."""
        if not token:
            return None
        
        user_id = None
        if self.is_mongo_connected and self._sessions_collection is not None:
            try:
                doc = self._sessions_collection.find_one({"token": token})
                if doc:
                    user_id = doc.get("user_id")
            except Exception:
                pass
        
        if not user_id:
            session_file = self._local_auth_path.parent / "_sessions.json"
            if session_file.exists():
                try:
                    sessions = json.loads(session_file.read_text(encoding="utf-8"))
                    if token in sessions:
                        user_id = sessions[token].get("user_id")
                except Exception:
                    pass
        
        if user_id:
            return self.get_user_by_id(user_id)
        return None

    def invalidate_session(self, token: str) -> bool:
        """Remove a session token on user logout."""
        if not token:
            return False
        
        if self.is_mongo_connected and self._sessions_collection is not None:
            try:
                self._sessions_collection.delete_many({"token": token})
            except Exception:
                pass
        
        session_file = self._local_auth_path.parent / "_sessions.json"
        if session_file.exists():
            try:
                sessions = json.loads(session_file.read_text(encoding="utf-8"))
                if token in sessions:
                    del sessions[token]
                    session_file.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
            except Exception:
                pass
        return True


def get_auth_service() -> AuthService:
    """Get singleton AuthService instance."""
    return AuthService.get_instance()
