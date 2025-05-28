# ---------- era_detector/Dockerfile ----------
# Purpose: lightweight image for era-detection (PELT, Bayesian CPD, Sticky-HDP-HMM)
# Notes  :
#   • Python 3.11 base for better perf & smaller wheels
#   • Uses `uv` for dependency resolution (fast-pip replacement)
#   • Relies on pinned versions in requirements_era_detection.txt so no manual patching is needed
#   • Only extra system package is git (required for VCS requirements)
#   • Entrypoint runs the detection script but can be overridden via docker-compose

FROM python:3.10-slim AS runtime

# Install Rust toolchain and build dependencies
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup component add rustfmt

# Install uv and maturin for Python package management and Rust builds
RUN pip install --no-cache-dir uv maturin

# workspace
WORKDIR /app

# copy requirement spec & install (layer-cached)
COPY requirements_era_detection.txt .
RUN uv pip install --system --no-cache -r requirements_era_detection.txt

# copy source
COPY run_era_detection.py .
COPY rust_hmm ./rust_hmm

# default command (can be overridden)
CMD ["python", "run_era_detection.py"] # Changed to CMD to be easily overridden by docker compose run for sanity checks
