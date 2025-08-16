# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DJANGO_SETTINGS_MODULE=core.settings

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # PostgreSQL client
    libpq-dev \
    # Build essentials for Python packages
    gcc \
    g++ \
    # Image processing libraries (updated for Debian Trixie)
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # OCR dependencies
    tesseract-ocr \
    # PDF processing
    poppler-utils \
    # Additional dependencies for Python packages
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-prod.txt

# Copy project files
COPY . .

# Copy entrypoint script and ensure it has proper permissions
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create necessary directories
RUN mkdir -p staticfiles static extracted_texts temp_extracts

# Collect static files (with WhiteNoise compression)
RUN python manage.py collectstatic --noinput

# Create a non-root user and fix permissions
RUN useradd -m -u 1000 django && \
    chown -R django:django /app && \
    chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8001

# Run with Gunicorn using the entrypoint script
CMD ["/app/entrypoint.sh"]