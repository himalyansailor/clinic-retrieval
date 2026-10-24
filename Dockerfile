FROM python:3.12-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.hf

# CPU torch: the encoder runs on CPU by design, and the GPU belongs to the LLM.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY clinic_retrieval ./clinic_retrieval
RUN pip install .

# Bake the embedding model into the image so the container answers offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY data ./data
ENV CLINIC_DATA=/app/data
EXPOSE 8000
CMD ["uvicorn", "clinic_retrieval.app:app", "--host", "0.0.0.0", "--port", "8000"]
