web: gunicorn app:app --log-file=- --timeout=300
worker1: celery -A app.celery worker -Ofair -n worker1.%h --concurrency 1
worker2: celery -A app.celery worker -Ofair -n worker2.%h --concurrency 1
worker3: celery -A app.celery worker -Ofair -n worker3.%h --concurrency 1
worker4: celery -A app.celery worker -Ofair -n worker4.%h --concurrency 1
worker5: celery -A app.celery worker -Ofair -n worker5.%h --concurrency 1
worker6: celery -A app.celery worker -Ofair -n worker6.%h --concurrency 1
worker7: celery -A app.celery worker -Ofair -n worker7.%h --concurrency 1
worker8: celery -A app.celery worker -Ofair -n worker8.%h --concurrency 1
worker9: celery -A app.celery worker -Ofair -n worker9.%h --concurrency 1
worker10: celery -A app.celery worker -Ofair -n worker10.%h --concurrency 1
tabooworker: celery -A app.celery worker -Q taboo-queue --concurrency 1 -n tabooworker.%h
beat: celery -A app.celery beat
