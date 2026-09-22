import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langgraph.store.memory import InMemoryStore

from app.memory import episodic


class EpisodicMemoryTests(unittest.TestCase):
    def test_recall_renders_structured_episodes_for_the_prompt(self) -> None:
        test_store = InMemoryStore()
        test_store.put(
            episodic.EPISODIC_NAMESPACE,
            "episode-1",
            {
                "kind": "Episode",
                "content": {
                    "target": "classifier",
                    "situation": "A prior inspection",
                    "lesson": "Preserve uncertainty",
                },
            },
        )

        with patch.object(episodic, "store", test_store), patch.object(
            episodic, "EMBED_MODEL", ""
        ):
            context = episodic.recall_context("inspect classifier")

        self.assertIn("A prior inspection", context)
        self.assertIn("Preserve uncertainty", context)
        self.assertIn("historical experience", context)

    def test_save_and_load_round_trip(self) -> None:
        source_store = InMemoryStore()
        source_store.put(
            episodic.EPISODIC_NAMESPACE,
            "episode-1",
            {"content": {"target": "classifier", "lesson": "use strip"}},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            with patch.object(episodic, "store", source_store):
                episodic.save(path)

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved[0]["key"], "episode-1")

            restored_store = InMemoryStore()
            with patch.object(episodic, "store", restored_store):
                loaded = episodic.load(path)
                restored = restored_store.get(
                    episodic.EPISODIC_NAMESPACE, "episode-1"
                )

        self.assertEqual(loaded, 1)
        self.assertEqual(restored.value["content"]["lesson"], "use strip")

    @patch("app.memory.episodic.create_memory_store_manager")
    def test_extraction_uses_structured_episode_manager(self, create: Mock) -> None:
        manager = create.return_value
        trajectory = {"user_request": "inspect model", "final_report": "safe"}

        episodic.extract_episode(trajectory, model=Mock())

        self.assertEqual(
            create.call_args.kwargs["namespace"], episodic.EPISODIC_NAMESPACE
        )
        self.assertEqual(create.call_args.kwargs["schemas"], [episodic.Episode])
        extraction_prompt = manager.invoke.call_args.args[0]["messages"][0]["content"]
        self.assertIn("inspect model", extraction_prompt)
        self.assertIn("safe", extraction_prompt)


if __name__ == "__main__":
    unittest.main()
