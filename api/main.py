"""The HTTP API: how other apps talk to the chatbot."""
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bots import tutor
from engine import config
from engine.llm import LLMError, Message, OllamaClient

app = FastAPI(title="Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = OllamaClient(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": config.OLLAMA_MODEL}


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    history: list[Message] = [{"role": "system", "content": tutor.SYSTEM_PROMPT}]
    history += [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        reply = llm.chat(history, temperature=tutor.TEMPERATURE)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return ChatResponse(reply=reply)
