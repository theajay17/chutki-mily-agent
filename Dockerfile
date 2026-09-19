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
COPY agent.py .

# Create non-root user for security  
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port for Railway
EXPOSE 8080
ENV PORT=8080

# Add health check endpoint

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/', timeout=5)"

# Run the agent in production mode
CMD ["python", "agent.py", "start"]
