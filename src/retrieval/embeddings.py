"""Embedding backends for dense retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic bag-of-tokens embedder for tests / offline CI."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            if not tokens:
                vectors.append(vec)
                continue
            for tok in tokens:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # probe dimension
        probe = self._model.encode(["dimension probe"], normalize_embeddings=True)
        self.dim = int(np.asarray(probe).shape[-1])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()


def get_embedder(model_name: str, use_hash: bool = False, hash_dim: int = 64) -> Embedder:
    if use_hash or model_name.lower() in {"hash", "fake", "test"}:
        return HashEmbedder(dim=hash_dim)
    return SentenceTransformerEmbedder(model_name)
