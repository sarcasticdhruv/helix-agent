"""
helix/models/embedder.py  —  Embedding providers for semantic search.
"""

from __future__ import annotations

import os

from helix.interfaces import EmbeddingProvider


class OpenAIEmbedder(EmbeddingProvider):
    """Embeddings via OpenAI text-embedding-3-small (1536 dims, cheap, fast)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("pip install openai")
        return self._client

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        try:
            response = await client.embeddings.create(model=self._model, input=texts)
            return [item.embedding for item in response.data]
        except Exception:
            return [[0.0] * self.dimensions for _ in texts]

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0] if results else [0.0] * self.dimensions

    @property
    def dimensions(self) -> int:
        return {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }.get(self._model, 1536)


class NullEmbedder(EmbeddingProvider):
    """Returns zero vectors. Use for tests or when no embedding key is available."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimensions for _ in texts]

    async def embed_one(self, text: str) -> list[float]:
        return [0.0] * self.dimensions

    @property
    def dimensions(self) -> int:
        return 4  # Small for tests so cosine similarity is fast
