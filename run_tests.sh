#!/bin/bash

# Run application tests using a custom environment file.
# Alternatively, use a command like 'heroku local -f Procfile.test'.
# Sample usage: ./run_tests.sh .env

env $(cat $1 | xargs) python -m unittest discover
