"""Database maintenance: python -m app.memory --help."""

import argparse
import json
from pathlib import Path

from app.memory.backend import EMBED_MODEL, MEMORY_OWNERS, MEMORY_PATH, memory_store
from app.memory.episodic import episode_namespace, recall_context
from app.memory.transfer import export_records, import_records, read_records, validate_record_owner


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage PostgreSQL episodic memory")
    commands = parser.add_subparsers(dest="command", required=True)
    initializer = commands.add_parser(
        "init", help="Create tables and enable pgvector (run before the app)"
    )
    importer = commands.add_parser("import-json", help="Import legacy or exported episodes")
    importer.add_argument("path", nargs="?", type=Path, default=MEMORY_PATH)
    exporter = commands.add_parser(
        "export-json", help="Export the selected owner's project episodes"
    )
    exporter.add_argument("path", type=Path)
    reindexer = commands.add_parser(
        "reindex", help="Rebuild vectors with the configured embedding model"
    )
    search = commands.add_parser(
        "search", help="Check retrieval without running a chat agent"
    )
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=3)
    for command in (initializer, importer, exporter, reindexer, search):
        selection = command.add_mutually_exclusive_group()
        selection.add_argument(
            "--owner", "--specialist", dest="owner", choices=MEMORY_OWNERS,
            default="shared", help="Memory owner (default: shared)",
        )
        if command is initializer:
            selection.add_argument("--all", action="store_true", help="Initialize every owner's database")
    args = parser.parse_args()
    try:
        specialist = None if args.owner == "shared" else args.owner
        if args.command == "search":
            context = recall_context(
                args.query, specialist=specialist, limit=args.limit
            )
            print(context or "No episodes found.")
            return
        # Validate the whole input before opening a database connection.
        records = read_records(args.path) if args.command == "import-json" else None
        if records is not None:
            validate_record_owner(records, specialist)
        if args.command == "init":
            owners = MEMORY_OWNERS if args.all else (args.owner,)
            for owner in owners:
                selected = None if owner == "shared" else owner
                with memory_store(specialist=selected, initialize=True) as store:
                    store.search(episode_namespace(selected), limit=1)
                status = "enabled" if EMBED_MODEL else "disabled"
                print(f"{owner} memory database initialized. Vector search {status}.")
            return
        with memory_store(specialist=specialist) as store:
            if args.command == "import-json":
                inserted, skipped = import_records(store, records)
                print(
                    f"Imported and verified {inserted} episodes; "
                    f"skipped {skipped} identical records."
                )
            elif args.command == "export-json":
                records = export_records(store, namespace=episode_namespace(specialist))
                # Exclusive creation keeps an existing export from being overwritten.
                with args.path.open("x", encoding="utf-8") as output:
                    json.dump(records, output, indent=2, default=str)
                print(f"Exported {len(records)} episodes to {args.path}.")
            elif args.command == "reindex":
                if not EMBED_MODEL:
                    raise ValueError("Set MEMORY_EMBED_MODEL before reindexing")
                # Snapshot first: put changes the backend's updated_at ordering.
                records = export_records(store, namespace=episode_namespace(specialist))
                for record in records:
                    store.put(tuple(record["namespace"]), record["key"], record["value"])
                print(f"Reindexed {len(records)} episodes.")
    except Exception as exc:
        parser.exit(1, f"Memory command failed: {exc}\n")


if __name__ == "__main__":
    main()
