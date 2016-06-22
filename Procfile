web: gunicorn app:app --log-file=- --timeout=300
worker: celery -A app.celery worker
