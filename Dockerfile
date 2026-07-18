# Local, reproducible runtime for the SherlockML FastAPI service.
# The Streamlit control room uses the same image with a different Compose command.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY agents /app/agents
COPY api /app/api
COPY dashboard /app/dashboard
COPY ml /app/ml
COPY models /app/models
COPY simulator /app/simulator

RUN python -m pip install --upgrade pip \
    && python -m pip install .

EXPOSE 8788

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8788"]
