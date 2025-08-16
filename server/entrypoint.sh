#!/bin/sh

set -e

echo "Starting FileAuthAI Server..."

# Wait for database to be ready (optional, but recommended)
if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    while ! python -c "
import psycopg2
import os
from urllib.parse import urlparse
url = urlparse(os.environ.get('DATABASE_URL'))
try:
    conn = psycopg2.connect(
        database=url.path[1:],
        user=url.username,
        password=url.password,
        host=url.hostname,
        port=url.port
    )
    conn.close()
    exit(0)
except Exception as e:
    print(f'Database not ready: {e}')
    exit(1)
    " 2>/dev/null; do
        sleep 2
    done
    echo "Database is ready!"
fi

# Run migrations
echo "Running database migrations..."
python manage.py migrate --noinput

# Create superuser if it doesn't exist (optional)
# echo "from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.create_superuser('admin', 'admin@example.com', 'admin') if not User.objects.filter(username='admin').exists() else None" | python manage.py shell

# Start Gunicorn
echo "Starting Gunicorn server on port 8001..."
exec gunicorn core.wsgi:application \
    --bind 0.0.0.0:8001 \
    --workers ${GUNICORN_WORKERS:-4} \
    --threads ${GUNICORN_THREADS:-2} \
    --worker-class sync \
    --worker-tmp-dir /dev/shm \
    --access-logfile - \
    --error-logfile - \
    --log-level ${LOG_LEVEL:-info} \
    --timeout 120 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --keep-alive 5