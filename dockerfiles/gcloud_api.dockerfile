FROM google/cloud-sdk:latest

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc python3-venv libgl1-mesa-glx libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists

### SETTTING UP PROJECT
# Create a virtual environment
RUN python3 -m venv /opt/venv

# Ensure the virtual environment is activated and use it for further commands
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy files from local to docker image
WORKDIR /app
COPY requirements.txt /app/
COPY pyproject.toml /app/
COPY .dvc/config /app/.dvc/config
COPY data.dvc /app/
COPY .git /app/.git/
COPY data /app/data/
COPY configs /app/configs/

COPY src /app/src/

COPY README.md /app/README.md
COPY tasks.py /app/tasks.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

EXPOSE $PORT
CMD exec uvicorn --port $PORT --host 0.0.0.0 src.mlops_project_tcs.api:app
