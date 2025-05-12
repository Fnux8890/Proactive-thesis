FROM python:3.9-slim

WORKDIR /app

# Install uv
RUN pip install uv

COPY requirements.txt .
# Install dependencies using uv
RUN uv pip install --system --no-cache -r requirements.txt

COPY db_utils.py .
COPY extract_features.py .

# Config will be mounted via volume in docker-compose

# CMD will be provided by docker-compose, but if run directly, it would be:
CMD ["uv", "run", "python", "extract_features.py"]
