FROM python:3.14-slim

# Install tesseract-ocr (required for RC.pdf parsing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (better caching)
COPY webapp/pyproject.toml webapp/uv.lock ./webapp/

# Install dependencies
RUN cd webapp && uv sync --no-dev --frozen 2>/dev/null || cd webapp && uv sync --no-dev

# Copy application code and data
COPY webapp/ ./webapp/
COPY analysis/ ./analysis/

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["uv", "run", "--directory", "webapp", \
    "streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
