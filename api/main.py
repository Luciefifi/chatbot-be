"""The HTTP API: how other apps talk to the chatbot."""
import json
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


def resolve_conversation(conversation_id: str | None) -> str:
    """Return the conversation to use: a new one, or the existing one the client asked for."""
    if conversation_id is None:
        return storage.create_conversation(tutor.NAME)
    if storage.conversation_exists(conversation_id):
        return conversation_id
    raise HTTPException(status_code=404, detail="Conversation not found")


def build_history(conversation_id: str, message: str) -> list[Message]:
    """System prompt + saved messages + the new message."""
    history: list[Message] = [{"role": "system", "content": tutor.SYSTEM_PROMPT}]
    history += storage.get_messages(conversation_id)
    history.append({"role": "user", "content": message})
    return history


def event(data: dict) -> str:
    """One line of the stream: a JSON object followed by a newline."""
    return json.dumps(data) + "\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": config.OLLAMA_MODEL}


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = resolve_conversation(request.conversation_id)
    history = build_history(conversation_id, request.message)

    try:
        reply = llm.chat(history, temperature=tutor.TEMPERATURE)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    storage.add_message(conversation_id, "user", request.message)
    storage.add_message(conversation_id, "assistant", reply)
    return ChatResponse(conversation_id=conversation_id, reply=reply)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    conversation_id = resolve_conversation(request.conversation_id)
    history = build_history(conversation_id, request.message)

    def generate() -> Iterator[str]:
        yield event({"type": "start", "conversation_id": conversation_id})
        reply = ""
        try:
            for piece in llm.chat_stream(history, temperature=tutor.TEMPERATURE):
                reply += piece
                yield event({"type": "token", "content": piece})
        except LLMError as e:
            yield event({"type": "error", "message": str(e)})
            return

        storage.add_message(conversation_id, "user", request.message)
        storage.add_message(conversation_id, "assistant", reply)
        yield event({"type": "done"})

    return StreamingResponse(generate(), media_type="application/x-ndjson")
