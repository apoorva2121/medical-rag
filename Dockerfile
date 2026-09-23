FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir . anthropic openai
COPY config.yaml .
COPY src ./src
COPY artifacts ./artifacts
ENV PYTHONPATH=/app/src KMP_DUPLICATE_LIB_OK=TRUE
EXPOSE 8000
CMD ["uvicorn", "medrag.app:app", "--host", "0.0.0.0", "--port", "8000"]
