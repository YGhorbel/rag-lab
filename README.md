# RAG Lab

Local-first, evaluation-driven Retrieval-Augmented Generation workbench for a
synthetic e-commerce catalog. RAG is the product.

See [`RAG_Lab_Open_Source_Developer_Spec.md`](RAG_Lab_Open_Source_Developer_Spec.md)
for the full developer specification.

## Status (this pass)

| Phase | Status |
|-------|--------|
| 0 Hardware / config | Ready (model IDs + settings) |
| 1 Synthetic data + gold set | Ready |
| 2 Retrieval baselines | Ready |
| 3 Hybrid + filters (Qdrant) | Ready |
| 4 Rerank + context | Stub |
| 5 Grounded generation | Stub |
| 6 Streamlit workbench | Stubs + Dataset Explorer |
| 7 Graph / polish | Deferred |

## Requirements

- Python 3.11+
- Docker (for Qdrant; optional app container)
- Ollama Cloud API key for generation (`OLLAMA_API_KEY`)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# Start Qdrant
docker compose up -d qdrant

# Generate synthetic catalog + gold benchmark
rag-lab-generate-catalog
rag-lab-generate-benchmark

# Index (hash embeddings for offline) + evaluate
rag-lab-index --hash-embeddings
rag-lab-eval --mode hybrid --subset demo

# Tests
pytest

# UI
streamlit run app.py
```

Dataset Explorer works immediately after data generation. Other tabs are
placeholders until Phases 2–5 land.

### One-command demo stack

```bash
docker compose up -d --build
```

- Streamlit: http://localhost:8501
- Qdrant: http://localhost:6333

Generation uses Ollama Cloud (`https://ollama.com`). Create a key at
https://ollama.com/settings/keys and set `OLLAMA_API_KEY` in `.env`.
Default model is `gpt-oss:20b`. Cloud model names come from
`https://ollama.com/api/tags` (`gemma4:31b`, `gpt-oss:120b`, and others).

## Dataset defaults

- 2,000 products → `data/processed/catalog_v1.jsonl`
- 500 benchmark queries → `data/benchmarks/benchmark_v1.jsonl`
- 80 demo queries → `data/benchmarks/demo_v1.jsonl`
- Languages: English + French
- Deterministic seed: `42` (override via `.env`)

## Design notes

- Structured filters for hard constraints (price, stock, brand, category).
- Hybrid dense + lexical retrieval later; product = one retrieval unit.
- Evaluation against synthetic gold labels is a first-class feature.
- Generation goes through Ollama Cloud. Licenses: see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Hardware note (Phase 0)

Check RAM/disk before pulling embedding and reranker weights.
The generator runs on Ollama Cloud, so it does not need a local pull.

## Docker note

If `docker compose` fails with a missing `~/.docker/desktop/docker.sock`, point the CLI at the engine socket:

```bash
export DOCKER_HOST=unix:///var/run/docker.sock
docker compose up -d qdrant
```
