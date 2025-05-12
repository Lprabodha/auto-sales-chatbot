from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import predict_intent, get_response_with_suggestions
from mongo_service import save_feedback
from retrain_daily import retrain_from_feedback

app = FastAPI()


class ChatRequest(BaseModel):
    query: str


class FeedbackRequest(BaseModel):
    query: str
    response: str
    predicted_intent: str
    prob: float
    thumbs_up: bool


@app.post("/chat")
def chat_api(req: ChatRequest):
    intent, prob = predict_intent(req.query)
    response_text, suggestions = get_response_with_suggestions(intent, req.query)

    return {
        "response": response_text,
        "prob": round(prob, 2),
        "intent": intent,
        "suggestions": suggestions
    }


@app.post("/feedback")
def feedback_api(req: FeedbackRequest):
    save_feedback(
        query=req.query,
        response=req.response,
        predicted_intent=req.predicted_intent,
        prob=req.prob,
        thumbs_up=req.thumbs_up
    )
    return {"status": "feedback recorded"}


@app.post("/retrain-now")
def manual_retrain():
    retrain_from_feedback()
    return {"status": "Retraining completed (if applicable)."}