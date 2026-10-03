"""Citation attribution package for ResearchMate.

Provides grounded citation models, provenance tracking, and source snippet extraction.
"""

from src.citations.citation_engine import CitationEngine
from src.models.citation import Citation

__all__ = ["Citation", "CitationEngine"]
