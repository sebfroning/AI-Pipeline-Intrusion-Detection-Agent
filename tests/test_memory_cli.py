import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, call, patch

from langgraph.store.memory import InMemoryStore

from app.memory.__main__ import main
from app.memory.backend import MEMORY_OWNERS
from app.memory.episodic import EPISODIC_NAMESPACE, episode_namespace


class MemoryCliTests(unittest.TestCase):
    def test_init_all_initializes_every_database(self):
        with patch.object(sys, "argv", ["memory", "init", "--all"]), patch(
            "app.memory.__main__.memory_store", side_effect=lambda **_: nullcontext(InMemoryStore())
        ) as connect, patch("builtins.print"):
            main()
        self.assertEqual(connect.call_args_list, [
            call(specialist=None if owner == "shared" else owner, initialize=True)
            for owner in MEMORY_OWNERS
        ])

    def test_owner_import_and_export_use_private_store(self):
        store = InMemoryStore()
        store.put(EPISODIC_NAMESPACE, "shared", {"lesson": "exclude shared"})
        store.put(episode_namespace("weather"), "weather", {"lesson": "exclude weather"})
        original = [{"namespace": list(episode_namespace("fruit")), "key": "fruit", "value": {"lesson": "fruit lesson"}}]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "in.json", Path(directory) / "out.json"
            source.write_text(json.dumps(original))
            with patch("app.memory.__main__.memory_store", side_effect=lambda **_: nullcontext(store)) as connect, patch("builtins.print"):
                for command, path in [("import-json", source), ("export-json", output)]:
                    with patch.object(sys, "argv", ["memory", command, str(path), "--owner", "fruit"]):
                        main()
                self.assertEqual(connect.call_args_list, [call(specialist="fruit")] * 2)
            self.assertEqual(json.loads(output.read_text()), original)

    def test_wrong_owner_import_fails_before_connecting(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "in.json"
            source.write_text(json.dumps([
                {"namespace": list(episode_namespace(owner)), "key": owner, "value": {"lesson": owner}}
                for owner in ("fruit", "weather")
            ]))
            with patch.object(sys, "argv", ["memory", "import-json", str(source), "--owner", "fruit"]), patch(
                "app.memory.__main__.memory_store"
            ) as connect, patch("sys.stderr"):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 1)
            connect.assert_not_called()

    def test_reindex_touches_only_selected_owner(self):
        store = InMemoryStore()
        store.put(episode_namespace("fruit"), "fruit", {"lesson": "fruit"})
        store.put(episode_namespace("weather"), "weather", {"lesson": "weather"})
        tracked_store = Mock(wraps=store)
        with patch.object(sys, "argv", ["memory", "reindex", "--owner", "fruit"]), patch(
            "app.memory.__main__.memory_store", return_value=nullcontext(tracked_store)
        ) as connect, patch("app.memory.__main__.EMBED_MODEL", "test"), patch("builtins.print"):
            main()
        connect.assert_called_once_with(specialist="fruit")
        tracked_store.put.assert_called_once_with(episode_namespace("fruit"), "fruit", {"lesson": "fruit"})

    def test_search_preserves_specialist_option(self):
        for option in ("--owner", "--specialist"):
            with self.subTest(option=option), patch.object(sys, "argv", ["memory", "search", "apple", option, "fruit"]), patch(
                "app.memory.__main__.recall_context", return_value="episode"
            ) as recall, patch("builtins.print"):
                main()
                recall.assert_called_once_with("apple", specialist="fruit", limit=3)

    def test_import_then_export_preserves_legacy_content(self):
        store = InMemoryStore()
        original = [{
            "namespace": ["memories", "episodes"],
            "key": "original-id",
            "value": {"content": {"lesson": "preserve uncertainty"}},
        }]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "memory.json"
            output = Path(directory) / "export.json"
            source.write_text(json.dumps(original))
            with patch("app.memory.__main__.memory_store", side_effect=lambda **_: nullcontext(store)), patch("builtins.print"):
                for command, path in [("import-json", source), ("import-json", source), ("export-json", output)]:
                    with patch.object(sys, "argv", ["memory", command, str(path)]):
                        main()
            records = json.loads(output.read_text())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["key"], "original-id")
            self.assertEqual(records[0]["value"], original[0]["value"])
            self.assertEqual(records[0]["namespace"], list(EPISODIC_NAMESPACE))
            self.assertEqual(json.loads(source.read_text()), original)

    def test_invalid_import_exits_before_connecting(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.json"
            source.write_text("{}")
            with patch.object(sys, "argv", ["memory", "import-json", str(source)]), patch(
                "app.memory.__main__.memory_store"
            ) as connect, patch("sys.stderr"):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 1)
            connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
