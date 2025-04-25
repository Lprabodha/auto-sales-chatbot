# Auto Sales Chatbot (FastAPI + PyTorch + MongoDB)

A complete production-ready intelligent chatbot for vehicle sales, supporting:
- AI-powered intent detection (price queries, location, brand, year)
- Dynamic answers via MongoDB (cars + bikes database)
- Retraining from real user feedback automatically

---

## 📦 Project Structure

```
/auto_sales_chatbot/
├── ai_chatbot_dynamic.py
├── train_model.py
├── retrain_from_feedback.py
├── prepare_data_from_mongo.py
├── fetch_utils.py
├── feedback_tools.py
├── /data/
│    ├── intents.json
│    ├── mongo_generated_training_data.pkl
├── /model/
│    ├── vectorizer.pkl
│    ├── label_encoder.pkl
│    ├── intent_model.pt
│    ├── confusion_matrix.png
├── requirements.txt
├── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model initially
```bash
python train_model.py
```

### 3. Start the chatbot API
```bash
uvicorn ai_chatbot_dynamic:app --reload
```

- Open API documentation at: http://127.0.0.1:8000/docs

---

## 🔁 Retraining from User Feedback

### 1. After users interact and you collect feedback (likes 👍):
```bash
python retrain_from_feedback.py
```
- This updates `data/intents.json` and retrains the model automatically.


---

## 🔥 Optional: Generate more training data from MongoDB

```bash
python prepare_data_from_mongo.py
```

- This will create realistic patterns like "Toyota Aqua 2024" for training.

---

## 📊 Monitor Chatbot Performance

```bash
python feedback_tools.py
```
- See top feedback messages and summaries!


---

## 📚 Main Features

- ✅ Dynamic MongoDB-powered vehicle answers
- ✅ Smart fallback recovery if confidence low
- ✅ Buyer intent detection (price, budget, seller, location, model year)
- ✅ Train on real-world feedback
- ✅ Easy to extend new intents, patterns, and models

---

## ✉️ Need Help?
Feel free to contact the developer if you need assistance in production deployment, advanced intent tuning, or scaling your chatbot system! 🚀