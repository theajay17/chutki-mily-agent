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

# Bake the Silero VAD model into the image so workers never fetch it at runtime.
# Runs as 'app' on purpose: the cache lives in the user's home, so downloading as
# root would leave it where app cannot read it.
#
# Loads the model directly rather than going through `agent.py download-files`,
# because that only covers plugins the CLI knows how to prefetch.
#
# Non-fatal by design. agent.py loads the VAD lazily in a background thread with
# a timeout and never blocks a call on it, so a cold image costs slightly worse
# turn-taking on the first call instead of breaking calls.
RUN python -c "from livekit.plugins import silero; silero.VAD.load()" || \
    echo "WARN: VAD model not baked in; workers will fetch it in the background"

# Expose port for Railway
EXPOSE 8080
ENV PORT=8080

# Add health check endpoint

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/', timeout=5)"

# Run the agent in production mode
CMD ["python", "agent.py", "start"]
