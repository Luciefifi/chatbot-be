"""Talks to the language model. This is the only file that knows about Ollama."""
import json
from collections.abc import Iterator
from typing import Literal, TypedDict

import httpx


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(Exception):
    """Raised when the model can't be reached or returns an error."""


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def chat(self, messages: list[Message], temperature: float = 0.7) -> str:
        """Send the whole conversation and return the model's reply."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = self._http.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise LLMError("Can't reach Ollama. Is it running?") from e
        except httpx.TimeoutException as e:
            raise LLMError("The model took too long to answer.") from e
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama returned {e.response.status_code}: {e.response.text}") from e

        return response.json()["message"]["content"]

    def chat_stream(self, messages: list[Message], temperature: float = 0.7) -> Iterator[str]:
        """Send the whole conversation and yield the reply piece by piece as it's generated."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as response:
                if response.is_error:
                    response.read()
                    raise LLMError(f"Ollama returned {response.status_code}: {response.text}")

                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise LLMError(chunk["error"])
                    if chunk["done"]:
                        break
                    yield chunk["message"]["content"]
        except httpx.ConnectError as e:
            raise LLMError("Can't reach Ollama. Is it running?") from e
        except httpx.TimeoutException as e:
            raise LLMError("The model took too long to answer.") from e
