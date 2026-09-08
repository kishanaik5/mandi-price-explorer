FROM python:3.11-slim

# Install system dependencies & Nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user && \
    mkdir -p /var/log/nginx /var/lib/nginx /tmp && \
    chown -R user:user /var/log/nginx /var/lib/nginx /tmp && \
    chmod -R 777 /var/log/nginx /var/lib/nginx /tmp
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    fastapi uvicorn[standard] sse-starlette pydantic python-dotenv

# Copy application files
COPY --chown=user:user . .
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER user

EXPOSE 7860

CMD ["./entrypoint.sh"]
