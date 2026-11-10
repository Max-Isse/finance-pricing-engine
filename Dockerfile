FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# The extra index provides the CPU-only PyTorch wheel, which is much smaller than the default build.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

EXPOSE 8000

# Train a first model if none exists, then serve it.
CMD ["sh", "-c", "[ -f artifacts/champion.pt ] || python pipelines/weekly_retrain.py; uvicorn src.service.app:app --host 0.0.0.0 --port 8000"]
