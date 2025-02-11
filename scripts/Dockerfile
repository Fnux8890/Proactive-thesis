FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
LABEL maintainer="Your Name <your_email@example.com>"

# Set the working directory to /app.
WORKDIR /app

# Copy your application files.
COPY requirements.txt pdf_to_markdown.py ./

# Ensure the script is executable.
RUN chmod +x pdf_to_markdown.py

# Create a Python virtual environment using uv.
RUN uv venv

# Activate the virtual environment and install dependencies.
RUN bash -c "source .venv/bin/activate && uv pip install -r requirements.txt --verbose"

# Create the folders for input and output.
RUN mkdir -p /app/data /app/output

# Declare volumes so that these directories can be bind-mounted.
VOLUME ["/app/data", "/app/output"]

# Optionally, list /app for verification.
RUN ls -la /app

# Ensure that the virtual environment is in the PATH.
ENV PATH="/app/.venv/bin:$PATH"

# Set the container entrypoint to run your script via uv.
ENTRYPOINT ["uv", "run", "pdf_to_markdown.py"]
CMD ["--help"]