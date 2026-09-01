"""LangMem-backed episodic and semantic memory for the agent pipeline.

Two namespaces are kept separate so recall can be scoped:

- semantic  : durable facts ("resnet18_poison.pth is a BadNets test fixture")
- episodic  : what happened during a past run ("2026-09-01 scan: strip flagged
              the model, mmbd agreed")

Both are stored in a LangGraph ``BaseStore``. ``InMemoryStore`` is ephemeral, so
this module also persists the store to JSON between CLI runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

SEMANTIC_NAMESPACE = ("memories", "semantic")
EPISODIC_NAMESPACE = ("memories", "episodic")

_REPO_ROOT = Path(__file__).resolve().parents[2]
MEMORY_PATH = Path(
    os.getenv("MEMORY_STORE_PATH", str(_REPO_ROOT / "data" / "memory.json"))
)

# nomic-embed-text is 768-dim; override both together if you use another model.
EMBED_MODEL = os.getenv("MEMORY_EMBED_MODEL", "nomic-embed-text")
EMBED_DIMS = int(os.getenv("MEMORY_EMBED_DIMS", "768"))


def _build_index() -> dict[str, Any] | None:
    """Semantic-search index config, or None to disable vector search."""
    if not EMBED_MODEL:
        return None

    from langchain_ollama import OllamaEmbeddings

    return {"dims": EMBED_DIMS, "embed": OllamaEmbeddings(model=EMBED_MODEL)}


def build_store() -> InMemoryStore:
    return InMemoryStore(index=_build_index())


store = build_store()


def _memory_tools(namespace: tuple[str, ...], label: str, instructions: str):
    return [
        create_manage_memory_tool(
            namespace=namespace,
            store=store,
            name=f"manage_{label}_memory",
            instructions=instructions,
        ),
        create_search_memory_tool(
            namespace=namespace,
            store=store,
            name=f"search_{label}_memory",
        ),
    ]


SEMANTIC_TOOLS = _memory_tools(
    SEMANTIC_NAMESPACE,
    "semantic",
    "Record durable facts that stay true across runs: model identities and "
    "paths, dataset and architecture details, defense compatibility "
    "constraints, and user preferences. Do not record one-off run results.",
)

EPISODIC_TOOLS = _memory_tools(
    EPISODIC_NAMESPACE,
    "episodic",
    "Record what happened in a specific run: which defenses were executed, the "
    "verdicts they returned, notable errors, and how long they took. Include "
    "the model reference and a timestamp so the episode can be found later.",
)

MEMORY_TOOLS = [*SEMANTIC_TOOLS, *EPISODIC_TOOLS]


def recall(query: str, *, namespace: tuple[str, ...], limit: int = 5) -> list[str]:
    """Fetch memories for prompt injection without going through the LLM."""
    items = store.search(namespace, query=query, limit=limit)
    return [str(item.value.get("content", item.value)) for item in items]


def recall_context(query: str, limit: int = 3) -> str:
    """Render both memory types as a prompt block. Empty string when nothing."""
    facts = recall(query, namespace=SEMANTIC_NAMESPACE, limit=limit)
    episodes = recall(query, namespace=EPISODIC_NAMESPACE, limit=limit)
    if not facts and not episodes:
        return ""

    sections = []
    if facts:
        sections.append("Known facts:\n" + "\n".join(f"- {f}" for f in facts))
    if episodes:
        sections.append("Past runs:\n" + "\n".join(f"- {e}" for e in episodes))
    return "<memories>\n" + "\n\n".join(sections) + "\n</memories>"


def save(path: Path | None = None) -> None:
    """Persist every memory to JSON so it survives process restarts."""
    target = path or MEMORY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for namespace in (SEMANTIC_NAMESPACE, EPISODIC_NAMESPACE):
        for item in store.search(namespace, limit=1000):
            records.append(
                {
                    "namespace": list(namespace),
                    "key": item.key,
                    "value": item.value,
                }
            )

    target.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def load(path: Path | None = None) -> int:
    """Restore memories from JSON. Returns how many were loaded."""
    source = path or MEMORY_PATH
    if not source.exists():
        return 0

    records = json.loads(source.read_text(encoding="utf-8"))
    for record in records:
        store.put(tuple(record["namespace"]), record["key"], record["value"])
    return len(records)
