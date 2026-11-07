FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

COPY . .

EXPOSE 8000
CMD ["uvicorn", "voyageiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
