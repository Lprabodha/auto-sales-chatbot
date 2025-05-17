
# 🚗 AutoBot – Intelligent Chatbot for Auto Sales
![Python](https://img.shields.io/badge/python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-green)
![Open Source](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-EE4C2C?logo=pytorch)
![MongoDB](https://img.shields.io/badge/MongoDB-Integrated-brightgreen?logo=mongodb)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Made With Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)


AutoBot is an AI-powered chatbot built using **PyTorch**, **FastAPI**, and **MongoDB** that helps users search and filter vehicles intelligently. It can handle queries related to brands, models, fuel types, price ranges, and more, and supports feedback collection for continuous improvement via scheduled model retraining.

---

## 🚀 Features

- 🔍 **Natural Language Understanding** for vehicle queries (e.g., *“Show me Toyota Vitz under 5 million”*)
- 📊 **Price, model, brand, and type filters** with fuzzy matching
- 💬 **Interactive Chat Mode** or REST API
- 🧠 **ML Model Training** with intent classification using PyTorch
- ♻️ **Daily Retraining from Feedback** (admin-reviewed)
- 📈 **Feedback Logging** with thumbs up/down reactions
- 📂 **MongoDB Integration** for vehicle data and feedback history

---

## 🧰 Tech Stack

- **Python 3.8+**
- **PyTorch**
- **FastAPI**
- **MongoDB**
- **Uvicorn** (for dev server)
- **scikit-learn** (metrics)
- **matplotlib** (for training metrics)

---

## 📁 Folder Structure

```
auto_sales_bot/
│
├── data/
│   └── intents.json              # Training intents
├── models/
│   └── model.pth                 # Saved PyTorch model
├── utils/
│   └── preprocessing.py          # Preprocessing utils
├── api.py                        # FastAPI endpoint
├── chatbot.py                    # Chat logic & ML inference
├── train.py                      # Training script
├── retrain_daily.py              # Scheduled retraining
├── mongo_service.py              # MongoDB interface
└── requirements.txt              # Dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/auto_sales_bot.git
cd auto_sales_bot
```

### 2. Create & activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start MongoDB

Ensure MongoDB is running locally at `mongodb://localhost:27017`. Use Docker or install MongoDB natively.

  #### Sample Data & Retraining

    🚗 Sample vehicle listings JSON: data/sample_vehicles.json

  #### Import Sample Data to MongoDB

  ```
  mongoimport --uri "mongodb://localhost:27017" \
  --db auto_sales_bot \
  --collection vehicles \
  --file data/sample_vehicles.json \
  --jsonArray
  ```

### 5. Configure your .env file
```
MONGO_URI=mongodb://localhost:27017
DB_NAME=auto_sales_bot
MODEL_PATH=models/model.pth
SECRET_KEY=your-secret-key
```

### 6. Train the model (initial run)

```bash
python train.py
```

### 7. Start the API

```bash
uvicorn api:app --reload
```

---

## 📬 API Endpoints

### `POST /chat`

**Request:**
```json
{ "query": "Show me Toyota Vitz under 5 million" }
```

**Response:**
```json
{
  "response": "Here are some Toyota Vitz available:",
  "prob": 0.97,
  "intent": "ask_brand_model",
  "suggestions": [
    {
      "id": 1,
      "model_name": "Vitz 2019",
      "vehicle_name": "Toyota Vitz",
      "year": 2019,
      "price": 9500000,
      "mileage": 60000
    }
  ]
}
```

### `POST /feedback`

**Request:**
```json
{
  "query": "Any Toyota cars under 4 million?",
  "response": "Here are some Toyota available:",
  "predicted_intent": "ask_price_range",
  "prob": 0.91,
  "thumbs_up": false
}
```

### `POST /retrain-now`

Forces model retraining based on feedback data (after admin update).

---

## ⏰ Automating Daily Retraining

Use a cron job or scheduler to run the retrain script:

```bash
0 0 * * * cd /path/to/project && .venv/bin/python retrain_daily.py
```

This will:
- Update `intents.json` from admin-approved feedback
- Retrain the model if required
- Save the updated model

---

## ✨ Contributions

Feel free to open issues or PRs for new features, model improvements, or bug fixes.



---

## 📜 License

MIT License. See `LICENSE` file for details.
