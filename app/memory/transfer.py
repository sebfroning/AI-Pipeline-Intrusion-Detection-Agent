"""Explicit, repeatable JSON migration and paginated memory export."""

from __future__ import annotations

import json
from pathlib import Path

from langgraph.store.base import BaseStore

from app.memory.episodic import EPISODIC_NAMESPACE, episode_namespace


def read_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Memory JSON must contain a list of records")
    normalized = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each memory record must be an object")
        namespace = record.get("namespace")
        key = record.get("key")
        if not isinstance(namespace, list) or not all(
            isinstance(x, str) for x in namespace
        ):
            raise ValueError("Each namespace must be a list of strings")
        namespace = tuple(namespace)
        if namespace == ("memories", "episodes"):
            namespace = EPISODIC_NAMESPACE
        elif namespace == EPISODIC_NAMESPACE:
            pass
        elif len(namespace) == 5 and namespace == episode_namespace(namespace[-1]):
            pass
        else:
            raise ValueError(
                f"Namespace is outside this project's episode collections: {namespace}"
            )
        if (
            not isinstance(key, str) or not key
            or not isinstance(record.get("value"), dict)
        ):
            raise ValueError("Each record needs a nonempty key and an object value")
        identity = (namespace, key)
        if identity in seen:
            raise ValueError(f"Duplicate memory key in import: {key}")
        seen.add(identity)
        normalized.append({"namespace": namespace, "key": key, "value": record["value"]})
    return normalized


def import_records(store: BaseStore, records: list[dict]) -> tuple[int, int]:
    """Preserve keys and values, skip identical records, reject overwrites."""
    pending = []
    skipped = 0
    # Detect conflicts before writing anything. Run migration with agents stopped.
    for record in records:
        existing = store.get(record["namespace"], record["key"])
        if existing is None:
            pending.append(record)
        elif existing.value == record["value"]:
            skipped += 1
        else:
            raise ValueError(
                f"Import would overwrite different content for key {record['key']}"
            )
    for record in pending:
        store.put(record["namespace"], record["key"], record["value"])
        saved = store.get(record["namespace"], record["key"])
        if saved is None or saved.value != record["value"]:
            raise RuntimeError(f"Import verification failed for key {record['key']}")
    return len(pending), skipped


def validate_record_owner(records: list[dict], specialist: str | None) -> None:
    """Reject mixed-owner imports before opening the destination database."""
    namespace = episode_namespace(specialist)
    if any(tuple(record["namespace"]) != namespace for record in records):
        raise ValueError(f"Import contains records outside owner {specialist or 'shared'}")


def export_records(
    store: BaseStore, *, page_size: int = 500, namespace: tuple[str, ...] | None = None
) -> list[dict]:
    """Enumerate project episodes, optionally restricted to one exact namespace."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    records = []
    offset = 0
    while True:
        items = store.search(namespace or EPISODIC_NAMESPACE[:-1], limit=page_size, offset=offset)
        records.extend(
            {"namespace": list(item.namespace), "key": item.key, "value": item.value}
            for item in items
            if namespace is None or item.namespace == namespace
        )
        if len(items) < page_size:
            return records
        offset += len(items)
