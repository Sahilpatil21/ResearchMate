"""Authentication and user management package for ResearchMate."""

from src.auth.auth_service import AuthService, get_auth_service
from src.auth.user_model import User

__all__ = ["AuthService", "User", "get_auth_service"]
