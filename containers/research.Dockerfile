FROM python:3.11.11-slim-bookworm
WORKDIR /workspace
COPY requirements/ requirements/
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements/language.txt -r requirements/vision.txt
COPY . .
RUN pip install --no-cache-dir --no-deps -e .
CMD ["python", "-m", "tn_compression", "--help"]
