Step-by-Step: Run Your Chatbot API
📁 Prerequisites
Make sure you're in your project directory

Your environment is activated (if using virtualenv)

Install all dependencies:


pip install -r requirements.txt

▶️ Run the Chatbot API

uvicorn chatbot_api:app --reload

If you're using the upgraded dynamic version:

uvicorn ai_chatbot_dynamic:app --reload

curl --location 'http://127.0.0.1:8000/chat' \
--header 'Content-Type: application/json' \
--data '{"user_id": "u1", "message": "Dio bike"}'