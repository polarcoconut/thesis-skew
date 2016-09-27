web: gunicorn app:app --log-file=- --timeout=300
worker1: celery -A app.celery worker -Ofair -n worker1.%h --concurrency 1
worker2: celery -A app.celery worker -Ofair -n worker2.%h --concurrency 1
tabooworker: celery -A app.celery worker -Q taboo-queue --concurrency 1 -n tabooworker.%h
beat: celery -A app.celery beat
