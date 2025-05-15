#!/bin/bash
service cron start

tail -f /dev/null

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
