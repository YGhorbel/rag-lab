# Third-Party Notices

RAG Lab depends on open-source libraries and (optionally) open-weight models.
Verify each model card and license revision before redistribution.

## Application dependencies

| Component | License (typical) | Notes |
|-----------|-------------------|-------|
| Python | PSF | Runtime |
| Pydantic / pydantic-settings | MIT | Schemas and settings |
| pandas | BSD-3-Clause | Tabular processing |
| Streamlit | Apache-2.0 | Demo UI |
| httpx | BSD-3-Clause | HTTP client |
| pytest | MIT | Tests |
| Qdrant (server image) | Apache-2.0 | Vector / hybrid index |
| qdrant-client | Apache-2.0 | Optional Phase 2+ |
| rank-bm25 | Apache-2.0 | Optional Phase 2+ |
| sentence-transformers | Apache-2.0 | Optional Phase 2+ |
| Ragas | Apache-2.0 | Optional secondary eval |

## Model weights (downloaded on first use)

| Model | Default ID | Intended use |
|-------|------------|--------------|
| Embedding | `Qwen/Qwen3-Embedding-0.6B` | Dense retrieval |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder rerank |
| Generator | `gpt-oss:20b` (Ollama Cloud) | Grounded answers |

"Open weights" is not the same as OSI-approved open source. Check the exact
Hugging Face / Ollama model card for the revision you pull.

## Policy

- Embeddings and reranking stay local. Generation uses Ollama Cloud.
- Cache embedding and reranker weights locally after the first download.
- Record any license changes when bumping model revisions.
