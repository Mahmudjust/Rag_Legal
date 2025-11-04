# ---- Base image ----
FROM python:3.11-slim

# ---- System packages (poppler-utils for pdftotext) ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- App code ----
COPY main.py .

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=false"]
