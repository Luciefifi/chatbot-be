"""A terminal chatbot for testing the engine."""
from bots import tutor
from engine import config
from engine.llm import LLMError, Message, OllamaClient


def main() -> None:
    llm = OllamaClient(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
    history: list[Message] = [{"role": "system", "content": tutor.SYSTEM_PROMPT}]

    print(f"Study tutor ({config.OLLAMA_MODEL}). Type 'quit' to exit.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue

        history.append({"role": "user", "content": user_text})
        try:
            reply = llm.chat(history, temperature=tutor.TEMPERATURE)
        except LLMError as e:
            print(f"[error] {e}\n")
            history.pop()  # drop the message that failed, so history stays valid
            continue

        history.append({"role": "assistant", "content": reply})
        print(f"Tutor: {reply}\n")


if __name__ == "__main__":
    main()
