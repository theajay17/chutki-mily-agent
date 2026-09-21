# Dockerfile for CHUTKI Mily Voice Agent
# Production deployment on Railway

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent.py call_context.py ./

# Create non-root user for security  
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Bake the Silero VAD model into the image so the first call of a cold worker
# does not pay the download. Runs as 'app' on purpose: the model cache lives in
# the user's home, so downloading as root would leave it where app cannot read
# it. Deliberately non-fatal — this is only a warm-up. If it fails the agent
# fetches the model at startup, and load_vad() in agent.py already degrades to
# STT-only endpointing rather than failing the call.
RUN python agent.py download-files || \
    echo "WARN: VAD model prewarm skipped; will be fetched at runtime"

# Expose port for Railway
EXPOSE 8080
ENV PORT=8080

# Add health check endpoint

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/', timeout=5)"

# Run the agent in production mode
CMD ["python", "agent.py", "start"]
