FROM python:3.11-slim AS base

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists

# Copy files from local to docker image
WORKDIR /app
COPY requirements.txt /app/
COPY pyproject.toml /app/
COPY .dvc /app/.dvc/
COPY data.dvc /app/
COPY .git /app/.git/

COPY src /app/src/

COPY README.md /app/README.md
COPY tasks.py /app/tasks.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose
RUN invoke prepare-data
