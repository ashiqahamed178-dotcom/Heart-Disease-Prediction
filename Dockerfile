# Serves the Streamlit app on 8501 by default.
# For the API instead:  docker run -p 8000:8000 heartrisk uvicorn heartrisk.api:app --host 0.0.0.0 --port 8000
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HEARTRISK_HOME=/app

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY app ./app
COPY models ./models
COPY reports ./reports

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
