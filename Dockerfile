# Multi-stage Streamlit app image for RAG Lab demos
FROM python:3.11-slim AS deps
WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
COPY pages ./pages
RUN pip install --no-cache-dir -e .

FROM python:3.11-slim AS runtime
WORKDIR /app
RUN groupadd --gid 1001 raglab \
    && useradd --uid 1001 --gid raglab --create-home --shell /bin/bash raglab \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --chown=raglab:raglab . .
USER 1001
EXPOSE 8501
ENV QDRANT_URL=http://qdrant:6333 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
