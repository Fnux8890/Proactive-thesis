# Use an official Python runtime as a parent image
# Use a specific version for reproducibility, e.g., 3.11 or 3.12
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv using pip
RUN pip install uv

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt using uv
# --system flag installs into the global site-packages, suitable for containers
# --no-cache prevents caching, reducing image size slightly
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run on container start using uv run
# Execute pipeline.py as a module within the pipeline_pkg package
CMD ["uv", "run", "-m", "pipeline_pkg.pipeline"] 