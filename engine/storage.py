"""Saves conversations in a SQLite database file."""
"""Saves conversations in a SQLite database file."""
import sqlite3
import uuid

from engine.llm import Message

DB_PATH = "chatbot.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    bot        TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = connect()
    conn.executescript(SCHEMA)
    conn.close()


def create_conversation(bot: str) -> str:
    conversation_id = str(uuid.uuid4())
    conn = connect()
    conn.execute("INSERT INTO conversations (id, bot) VALUES (?, ?)", (conversation_id, bot))
    conn.commit()
    conn.close()
    return conversation_id

def add_message(conversation_id: str, role: str, content: str) -> None:
    conn = connect()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content),
    )
    conn.commit()
    conn.close()

def get_messages(conversation_id: str) -> list[Message]:
    conn = connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return [{"role": row["role"], "content": row["content"]} for row in rows]

def conversation_exists(conversation_id: str) -> bool:
    conn = connect()
    row = conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    conn.close()
    return row is not None
