#!/bin/bash

echo "🚀 Starting Auto Sales Chatbot Setup..."

echo "🔹 Activating Python virtual environment..."
source venv/bin/activate

echo "🔹 Preparing dynamic data from MongoDB (optional)..."
python prepare_data_from_mongo.py

echo "🔹 Training the AI model..."
python train_model.py

echo "🔹 Launching FastAPI server (Uvicorn)..."
uvicorn ai_chatbot_dynamic:app --reload

