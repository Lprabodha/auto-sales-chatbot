#!/bin/bash
service cron start

# python3 /src/main.py
python3 /src/train_predict_vehicle.py

tail -f /dev/null

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
