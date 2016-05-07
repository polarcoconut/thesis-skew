#!/bin/bash

# Run application using a custom environment file.
# Alternatively, use a command like 'heroku local'.
# Sample usage: ./run .env -b 0.0.0.0:8000.

env $(cat $1 | xargs) gunicorn app:app --timeout=3000 ${@:2}
