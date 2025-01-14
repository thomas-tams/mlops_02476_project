FROM python:3.11-slim AS base

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists

# Copy files from local to docker image
COPY src/ src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY data/ data/

RUN pip install --upgrade pip
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["uvicorn", "src/mlops_project_tcs/api:app", "--host", "0.0.0.0", "--port", "8000"]
