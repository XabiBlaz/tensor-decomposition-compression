FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TN_RUNS_DIR=/runs \
    TN_DATA_DIR=/data \
    TN_UPLOADS_DIR=/uploads
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/language.txt -r requirements/vision.txt
COPY pyproject.toml README.md LICENSE ./
COPY tn_compression/ tn_compression/
COPY compression/ compression/
RUN pip install --no-cache-dir --no-deps . \
    && useradd --uid 10001 --create-home appuser \
    && mkdir -p /runs /data /cache /uploads \
    && chown -R appuser:appuser /runs /data /cache /uploads
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health', timeout=3)"
CMD ["python", "-m", "tn_compression.ui", "--host", "0.0.0.0"]
