#!/bin/bash

# Health check script for Celery worker
# This script checks if the Celery worker is responding to ping

set -e

echo "Checking Celery worker health..."

# Try to ping the Celery worker
celery -A core inspect ping --timeout=10

if [ $? -eq 0 ]; then
    echo "Celery worker is healthy"
    exit 0
else
    echo "Celery worker is not responding"
    exit 1
fi