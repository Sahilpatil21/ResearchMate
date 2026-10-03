"""Local embedding model provider for ResearchMate using sentence-transformers.

Provides fast, local, zero-cost, API-key-free dense embeddings for research paper chunks and queries.
Default model: 'all-MiniLM-L6-v2' (384-dimensional dense vectors, optimized for semantic retrieval).
"""

from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


class EmbeddingModel:
    """Manages local sentence-transformer models with lazy loading and singleton caching."""

    _instance: Optional[EmbeddingModel] = None

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the embedding model manager.

        Args:
            model_name: HuggingFace model identifier or local path.
                Defaults to EMBEDDING_MODEL_NAME environment variable or 'all-MiniLM-L6-v2'.
        """
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        )
        self._model = None
        self._dimension: Optional[int] = None

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> EmbeddingModel:
        """Get or create singleton instance of EmbeddingModel."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model on first usage."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = SentenceTransformer(self.model_name, device=device)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load sentence-transformers model '{self.model_name}': {e}"
                ) from e

    @property
    def model(self):
        """Access underlying SentenceTransformer model, loading if needed."""
        self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of the embedding model."""
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> List[List[float]]:
        """Compute dense vector embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Mini-batch size for inference.
            normalize: If True, L2-normalizes vectors (enables cosine similarity via dot product).
            show_progress_bar: Whether to display a progress bar during encoding.

        Returns:
            List of float vectors, each of length `dimension`.
        """
        if not texts:
            return []

        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str, normalize: bool = True) -> List[float]:
        """Compute dense vector embedding for a single search query string."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        self._load_model()
        embedding = self._model.encode(
            query.strip(),
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embedding.tolist()


def get_embedding_model(model_name: Optional[str] = None) -> EmbeddingModel:
    """Convenience accessor to get the singleton EmbeddingModel instance."""
    return EmbeddingModel.get_instance(model_name)
