"""The HTTP API: how other apps talk to the chatbot."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bots import tutor
from engine import config, storage
from engine.llm import LLMError, Message, OllamaClient

app = FastAPI(title="Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = OllamaClient(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
storage.init_db()


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": config.OLLAMA_MODEL}


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    if request.conversation_id is None:
        conversation_id = storage.create_conversation(tutor.NAME)
    elif storage.conversation_exists(request.conversation_id):
        conversation_id = request.conversation_id
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

    history: list[Message] = [{"role": "system", "content": tutor.SYSTEM_PROMPT}]
    history += storage.get_messages(conversation_id)
    history.append({"role": "user", "content": request.message})

    try:
        reply = llm.chat(history, temperature=tutor.TEMPERATURE)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    storage.add_message(conversation_id, "user", request.message)
    storage.add_message(conversation_id, "assistant", reply)
    return ChatResponse(conversation_id=conversation_id, reply=reply)
