from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb+srv://lprabodha1998:SfXnuKZIecrv3TUJ@cluster0.e2m4j.mongodb.net/")
db = client["vehicle_prices"]
history = db["chat_history"]

def top_feedback(limit=10):
    print("\n🔝 Top Feedback Messages:")
    cursor = history.find({"score": {"$gte": 1}}).sort("score", -1).limit(limit)
    for doc in cursor:
        print(f"User: {doc.get('user_id', 'unknown')} | Score: {doc.get('score', 0)}")
        print(f"Message: {doc.get('user_message', '')}")
        print(f"Intent: {doc.get('intent', '')} | Confidence: {doc.get('confidence', 0):.2f}")
        print("-")

def feedback_summary():
    total = history.count_documents({})
    positive = history.count_documents({"score": {"$gte": 1}})
    negative = history.count_documents({"score": {"$lt": 0}})
    print("\n📊 Feedback Summary:")
    print(f"Total Chats: {total}")
    print(f"Positive Feedback: {positive}")
    print(f"Negative Feedback: {negative}")

if __name__ == "__main__":
    feedback_summary()
    top_feedback(10)