"""User data models for authentication and multi-tenant access control."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class User(BaseModel):
    """Authenticated user account model."""

    user_id: str = Field(default_factory=lambda: f"usr_{uuid.uuid4().hex[:12]}")
    email: str
    full_name: str
    password_hash: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_login: Optional[str] = None

    def to_dict(self, include_hash: bool = False) -> Dict[str, Any]:
        """Convert User instance to dictionary."""
        d = {
            "user_id": self.user_id,
            "email": self.email.lower().strip(),
            "full_name": self.full_name.strip(),
            "created_at": self.created_at,
            "last_login": self.last_login,
        }
        if include_hash:
            d["password_hash"] = self.password_hash
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> User:
        """Create a User instance from a dictionary."""
        return cls(
            user_id=data.get("user_id") or f"usr_{uuid.uuid4().hex[:12]}",
            email=data.get("email", "").lower().strip(),
            full_name=data.get("full_name", "").strip(),
            password_hash=data.get("password_hash", ""),
            created_at=data.get("created_at") or datetime.now(timezone.utc).isoformat(),
            last_login=data.get("last_login"),
        )
