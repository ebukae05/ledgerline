FROM python:3.13-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Models and data are mounted at runtime, never baked into the image.
ENV PYTHONUNBUFFERED=1 WORKERS=1
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn ledgerline.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
