web: gunicorn app:app --log-file=- --timeout=300
worker: celery -A app.celery worker --concurrency 2
tabooworker: celery -A app.celery worker -Q taboo-queue --concurrency 1
beat: celery -A app.celery beat
