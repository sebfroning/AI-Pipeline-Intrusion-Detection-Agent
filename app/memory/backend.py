"""PostgreSQL connections and vector-index configuration.

Each memory operation owns its connection, including specialist calls running in
parallel. Importing the application never opens a database connection.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_ollama import OllamaEmbeddings
from langgraph.store.postgres import PostgresStore
from psycopg.conninfo import make_conninfo

from app.environment import REPO_ROOT


EMBED_MODEL = os.getenv("MEMORY_EMBED_MODEL", "nomic-embed-text")
EMBED_DIMS = int(os.getenv("MEMORY_EMBED_DIMS", "768"))
MEMORY_PROJECT = os.getenv("MEMORY_PROJECT", "intrusion-agent")
MEMORY_PATH = REPO_ROOT / "data" / "memory.json"
CONFIG_NAMESPACE = ("memory_configuration",)
CONFIG_KEY = "embedding"
SPECIALIST_PORTS = {"fruit": "5433", "weather": "5434", "mithridat": "5435"}
MEMORY_OWNERS = ("shared", *SPECIALIST_PORTS)


def validate_specialist(specialist: str | None) -> None:
    if specialist is not None and specialist not in SPECIALIST_PORTS:
        raise ValueError(f"Unknown memory specialist: {specialist!r}")


def connection_string(specialist: str | None = None) -> str:
    """Build libpq connection info, safely quoting special password characters."""
    validate_specialist(specialist)
    prefix = f"{specialist.upper()}_" if specialist is not None else ""
    url_key = f"{prefix}MEMORY_DATABASE_URL"
    if uri := os.getenv(url_key):
        return make_conninfo(uri, connect_timeout=5)
    password_key = f"{prefix}POSTGRES_PASSWORD"
    password = os.getenv(password_key)
    if not password or password == "replace-with-a-local-password":
        raise ValueError(f"Set {password_key} in .env or {url_key}")
    return make_conninfo(
        host=os.getenv(f"{prefix}POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv(f"{prefix}POSTGRES_PORT", SPECIALIST_PORTS.get(specialist, "5432")),
        dbname=os.getenv(f"{prefix}POSTGRES_DB", f"{specialist}_memory" if specialist else "agent_memory"),
        user=os.getenv(f"{prefix}POSTGRES_USER", f"{specialist}_agent" if specialist else "agent"),
        password=password,
        connect_timeout=5,
    )


def embedding_config() -> dict | None:
    if not EMBED_MODEL:
        return None
    if EMBED_DIMS <= 0:
        raise ValueError("MEMORY_EMBED_DIMS must be positive")
    return {
        "dims": EMBED_DIMS,
        "embed": OllamaEmbeddings(model=EMBED_MODEL),
        "fields": ["$"],
        "distance_type": "cosine",
        "ann_index_config": {"kind": "flat"},
    }


def _check_embedding_config(store: PostgresStore, *, initialize: bool) -> None:
    if not EMBED_MODEL:
        return
    expected = {"model": EMBED_MODEL, "dims": EMBED_DIMS}
    saved = store.get(CONFIG_NAMESPACE, CONFIG_KEY)
    if saved is None:
        if not initialize:
            raise ValueError("Initialize vector memory with: python -m app.memory init --all")
        store.put(CONFIG_NAMESPACE, CONFIG_KEY, expected, index=False)
    elif saved.value != expected:
        raise ValueError(
            "Embedding model/dimensions differ from this database. Restore the "
            "original settings or import an export into a new database with the "
            "new settings; existing vectors cannot be mixed with a new model."
        )


@contextmanager
def memory_store(
    *, specialist: str | None = None, initialize: bool = False
) -> Iterator[PostgresStore]:
    """Open a bounded-lifetime connection; only explicit init runs migrations."""
    with PostgresStore.from_conn_string(
        connection_string(specialist), index=embedding_config()
    ) as store:
        if initialize:
            store.setup()
        _check_embedding_config(store, initialize=initialize)
        yield store
