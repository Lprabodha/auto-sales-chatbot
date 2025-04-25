#!/bin/bash

# launch.sh (Auto Sales AI Chatbot)

echo "🚀 Starting Auto Sales Chatbot Setup..."

# Step 1: Activate your virtual environment
echo "🔹 Activating Python virtual environment..."
source venv/bin/activate

# Step 2: (Optional) Regenerate training data from MongoDB
echo "🔹 Preparing dynamic data from MongoDB (optional)..."
python prepare_data_from_mongo.py

# Step 3: Train the AI model
echo "🔹 Training the AI model..."
python train_model.py

# Step 4: Start FastAPI server
echo "🔹 Launching FastAPI server (Uvicorn)..."
uvicorn ai_chatbot_dynamic:app --reload

