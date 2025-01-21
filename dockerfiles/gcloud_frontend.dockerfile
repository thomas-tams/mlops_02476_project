FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*




RUN mkdir /app
WORKDIR /app

COPY requirements_frontend.txt /app/requirements_frontend.txt
COPY src/mlops_project_tcs/frontend.py /app/frontend.py

RUN pip install -r requirements_frontend.txt


ENV GOOGLE_APPLICATION_CREDENTIALS=/credential_secret/CREDENTIALS-CLOUD-RUN-VIEWER
EXPOSE $PORT
CMD exec streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0
