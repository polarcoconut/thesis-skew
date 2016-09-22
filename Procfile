web: gunicorn app:app --log-file=- --timeout=300
worker1: celery -A app.celery worker -Ofair -n worker1.%h
worker2: celery -A app.celery worker -Ofair -n worker2.%h
tabooworker: celery -A app.celery worker -Q taboo-queue --concurrency 1 -n tabooworker.%h
beat: celery -A app.celery beat
