FROM python:3.11-slim
WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=600
COPY requirements.txt .
COPY requirements-torch.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 600 --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt
RUN pip install --no-cache-dir --retries 10 --timeout 600 -r requirements.txt
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
