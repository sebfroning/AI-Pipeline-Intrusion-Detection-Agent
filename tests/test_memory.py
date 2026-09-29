import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.embeddings import Embeddings
from langgraph.store.memory import InMemoryStore

from app.memory import episodic
from app.memory.transfer import export_records, import_records, read_records


class TopicEmbeddings(Embeddings):
    """Predictable vectors to test ranking without contacting a model server."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0]


class EpisodicMemoryTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryStore()
        self.private_stores = {owner: InMemoryStore() for owner in ("fruit", "weather", "mithridat")}
        context = patch.object(
            episodic, "memory_store", side_effect=lambda *, specialist=None: nullcontext(
                self.store if specialist is None else self.private_stores[specialist]
            )
        )
        context.start()
        self.addCleanup(context.stop)
        embeddings = patch.object(episodic, "EMBED_MODEL", "")
        embeddings.start()
        self.addCleanup(embeddings.stop)

    def test_recall_renders_structured_episodes_for_the_prompt(self):
        self.store.put(episodic.EPISODIC_NAMESPACE, "episode-1", {
            "kind": "Episode", "content": {
                "target": "classifier", "situation": "A prior inspection",
                "lesson": "Preserve uncertainty",
            },
        })
        context = episodic.recall_context("inspect classifier")
        self.assertIn("A prior inspection", context)
        self.assertIn("Preserve uncertainty", context)
        self.assertIn("historical experience", context)

    def test_specialist_retrieval_includes_shared_but_not_other_specialists(self):
        for owner, lesson in [(None, "shared lesson"), ("fruit", "fruit lesson"), ("weather", "weather lesson")]:
            store = self.store if owner is None else self.private_stores[owner]
            store.put(episodic.episode_namespace(owner), "episode", {"lesson": lesson})
        # Old private records in the shared database must never be recalled.
        self.store.put(episodic.episode_namespace("fruit"), "old", {"lesson": "stale private lesson"})
        context = episodic.recall_context("inspect", specialist="fruit")
        self.assertIn("fruit lesson", context)
        self.assertIn("shared lesson", context)
        self.assertNotIn("weather lesson", context)
        self.assertNotIn("stale private lesson", context)
        supervisor_context = episodic.recall_context("inspect")
        self.assertIn("shared lesson", supervisor_context)
        self.assertNotIn("fruit lesson", supervisor_context)

    def test_vector_retrieval_ranks_similar_experience_first(self):
        self.store = InMemoryStore(index={"dims": 2, "embed": TopicEmbeddings()})
        self.store.put(episodic.EPISODIC_NAMESPACE, "rain", {"lesson": "rain forecast"})
        self.store.put(episodic.EPISODIC_NAMESPACE, "apple", {"lesson": "apple inspection"})
        with patch.object(episodic, "EMBED_MODEL", "test"):
            context = episodic.recall_context("apple", limit=1)
        self.assertIn("apple inspection", context)
        self.assertNotIn("rain forecast", context)

    def test_empty_collection_does_not_request_embeddings(self):
        embeddings = Mock(spec=Embeddings)
        self.store = InMemoryStore(index={"dims": 2, "embed": embeddings})
        with patch.object(episodic, "EMBED_MODEL", "test"):
            self.assertEqual(episodic.recall_context("inspect"), "")
        embeddings.embed_query.assert_not_called()
        embeddings.embed_documents.assert_not_called()

    def test_specialist_falls_back_to_shared_and_respects_limit(self):
        for index in range(5):
            self.store.put(episodic.EPISODIC_NAMESPACE, str(index), {"lesson": f"lesson-{index}"})
        context = episodic.recall_context("inspect", specialist="fruit", limit=2)
        self.assertEqual(context.count("Episode "), 2)

    @patch("app.memory.episodic.create_memory_store_manager")
    def test_extraction_uses_structured_episode_manager_in_requested_scope(self, create):
        trajectory = {"user_request": "inspect model", "final_report": "safe"}
        episodic.extract_episode(trajectory, model=Mock(), specialist="fruit")
        self.assertEqual(create.call_args.kwargs["namespace"], episodic.episode_namespace("fruit"))
        self.assertEqual(create.call_args.kwargs["schemas"], [episodic.Episode])
        self.assertIs(create.call_args.kwargs["store"], self.private_stores["fruit"])
        prompt = create.return_value.invoke.call_args.args[0]["messages"][0]["content"]
        self.assertIn("inspect model", prompt)
        self.assertIn("safe", prompt)

    def test_shared_outage_preserves_private_recall(self):
        self.private_stores["fruit"].put(episodic.episode_namespace("fruit"), "fruit", {"lesson": "fruit lesson"})

        def connect(*, specialist=None):
            if specialist is None:
                raise ConnectionError("shared unavailable")
            return nullcontext(self.private_stores[specialist])

        with patch.object(episodic, "memory_store", side_effect=connect), self.assertLogs(episodic.logger, "WARNING"):
            context = episodic.recall_context("inspect", specialist="fruit")
        self.assertIn("fruit lesson", context)

    def test_private_outage_still_fills_limit_from_shared(self):
        for index in range(3):
            self.store.put(episodic.EPISODIC_NAMESPACE, str(index), {"lesson": "shared lesson"})

        def connect(*, specialist=None):
            if specialist is not None:
                raise ConnectionError("private unavailable")
            return nullcontext(self.store)

        with patch.object(episodic, "memory_store", side_effect=connect), self.assertLogs(episodic.logger, "WARNING"):
            context = episodic.recall_context("inspect", specialist="fruit")
        self.assertEqual(context.count("Episode "), 3)

    def test_private_quota_reserves_shared_slot(self):
        for index in range(4):
            self.private_stores["fruit"].put(episodic.episode_namespace("fruit"), str(index), {"lesson": "private lesson"})
        self.store.put(episodic.EPISODIC_NAMESPACE, "shared", {"lesson": "shared lesson"})
        context = episodic.recall_context("inspect", specialist="fruit")
        self.assertEqual(context.count("private lesson"), 2)
        self.assertEqual(context.count("shared lesson"), 1)
        context = episodic.recall_context("inspect", specialist="fruit", limit=1)
        self.assertEqual(context.count("Episode "), 1)
        self.assertNotIn("shared lesson", context)

    def test_unknown_specialist_is_rejected_before_connecting(self):
        with patch.object(episodic, "memory_store") as connect:
            with self.assertRaisesRegex(ValueError, "Unknown"):
                episodic.recall_context("inspect", specialist="typo")
        connect.assert_not_called()


class MemoryTransferTests(unittest.TestCase):
    def test_legacy_import_preserves_content_and_is_repeatable(self):
        legacy = [{"namespace": ["memories", "episodes"], "key": "original-id", "value": {"content": {"lesson": "use strip"}}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            original = json.dumps(legacy)
            path.write_text(original)
            records = read_records(path)
            store = InMemoryStore()
            self.assertEqual(import_records(store, records), (1, 0))
            self.assertEqual(import_records(store, records), (0, 1))
            saved = store.get(episodic.EPISODIC_NAMESPACE, "original-id")
            self.assertEqual(saved.value, legacy[0]["value"])
            self.assertEqual(path.read_text(), original)

    def test_conflicting_import_is_rejected_before_any_write(self):
        store = InMemoryStore()
        namespace = episodic.EPISODIC_NAMESPACE
        store.put(namespace, "existing", {"lesson": "original"})
        records = [
            {"namespace": namespace, "key": "new", "value": {"lesson": "new"}},
            {"namespace": namespace, "key": "existing", "value": {"lesson": "different"}},
        ]
        with self.assertRaisesRegex(ValueError, "overwrite"):
            import_records(store, records)
        self.assertIsNone(store.get(namespace, "new"))

    def test_paginated_export_round_trip_covers_all_scopes(self):
        store = InMemoryStore()
        for index in range(7):
            namespace = episodic.episode_namespace("fruit" if index % 2 else None)
            store.put(namespace, str(index), {"lesson": str(index)})
        store.put(("unrelated",), "ignored", {"lesson": "ignored"})
        records = export_records(store, page_size=2)
        self.assertEqual(len(records), 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.json"
            path.write_text(json.dumps(records))
            restored = InMemoryStore()
            self.assertEqual(import_records(restored, read_records(path)), (7, 0))

    def test_invalid_namespace_fails_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps([{"namespace": ["other-project"], "key": "x", "value": {}}]))
            with self.assertRaisesRegex(ValueError, "outside"):
                read_records(path)


if __name__ == "__main__":
    unittest.main()
