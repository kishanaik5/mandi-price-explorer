#!/bin/bash
set -e

echo "Starting Streamlit UI on internal port 8501..."
streamlit run app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false &

echo "Starting FastAPI Backend on internal port 8000..."
uvicorn api:app --host 127.0.0.1 --port 8000 &

echo "Starting Nginx reverse-proxy on public port 7860..."
exec nginx -g "daemon off;" -c /app/nginx.conf
