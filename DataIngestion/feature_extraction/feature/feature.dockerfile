# syntax=docker/dockerfile:1.4

# Stage 1: Dependencies only (rebuild only when requirements.txt changes)
FROM nvcr.io/nvidia/rapidsai/base:25.04-cuda12.0-py3.10 AS dependencies

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Stage 2: Final image
FROM dependencies

WORKDIR /app

ENV USE_GPU=true

COPY *.py .

CMD ["python", "extract_features.py"]