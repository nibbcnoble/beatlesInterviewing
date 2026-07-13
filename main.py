from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.rag.interview_service import get_interview_service


app = FastAPI(title="Beatles Interview Chatbot")

PERSONAS = {
    "john": {
        "style": "darkly witty, goofy, sharp, lightly surreal, skeptical"
    },
    "paul": {
        "style": "upbeat, confident, polished, slightly domineering, energetic"
    },
    "george": {
        "style": "brooding, thoughtful, understated, reflective"
    },
    "ringo": {
        "style": "cheerful, friendly, warm, conversational"
    }
}

class InterviewRequest(BaseModel):
    beatle: str = Field(..., description="One of: john, paul, george, ringo")
    question: str = Field(..., min_length=3, max_length=1000)

class InterviewResponse(BaseModel):
    beatle: str
    question: str
    answer: str
    sources: list
    grounded: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/interview", response_model=InterviewResponse)
def interview(req: InterviewRequest):
    beatle = req.beatle.lower().strip()
    question = req.question.strip()

    if beatle not in PERSONAS:
        raise HTTPException(
            status_code=400,
            detail="Invalid beatle. Use john, paul, george, or ringo."
        )

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        service = get_interview_service()
        result = service.answer_question(beatle=beatle, question=question)

        return InterviewResponse(
            beatle=result["beatle"],
            question=question,
            answer=result["answer"],
            sources=[],
            grounded=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Interview generation failed: {str(e)}"
        )

