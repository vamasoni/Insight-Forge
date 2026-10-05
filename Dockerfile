FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml ./
COPY insightforge ./insightforge
COPY evals ./evals
COPY scripts ./scripts
COPY app ./app
RUN pip install ".[all]" && useradd -m forge && mkdir -p /app/runs /app/data && chown -R forge /app
USER forge
EXPOSE 8000 8501
CMD ["uvicorn", "insightforge.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
