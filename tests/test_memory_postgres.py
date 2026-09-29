"""Opt-in integration test against a dedicated PostgreSQL + pgvector database.

TEST_MEMORY_DATABASE_URL=postgresql://... python -m unittest discover -s tests -p test_memory_postgres.py -v
Creates LangGraph tables, uses a unique namespace, and removes its test records.
No Ollama server is required: deterministic vectors exercise real SQL ranking.
"""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from unittest.mock import patch

from langchain_core.embeddings import Embeddings

from app.memory import backend, episodic


class TestEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        coordinates = [1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0]
        return coordinates + [0.0] * (backend.EMBED_DIMS - 2)


@unittest.skipUnless(os.getenv("TEST_MEMORY_DATABASE_URL"), "Set TEST_MEMORY_DATABASE_URL for real PostgreSQL tests")
class PostgresMemoryTests(unittest.TestCase):
    def test_vectors_persist_across_connections_and_parallel_writes(self):
        namespace = ("integration_test", uuid4().hex)
        records = [("apple", {"lesson": "apple inspection"}), ("rain", {"lesson": "rain forecast"})]
        with patch.dict(os.environ, {"MEMORY_DATABASE_URL": os.environ["TEST_MEMORY_DATABASE_URL"]}), patch.object(
            backend, "OllamaEmbeddings", return_value=TestEmbeddings()
        ):
            self.assertTrue(backend.EMBED_MODEL, "Integration test requires vector search enabled")
            with backend.memory_store(initialize=True):
                pass
            try:
                def write(record):
                    with backend.memory_store() as store:
                        store.put(namespace, record[0], record[1])

                with ThreadPoolExecutor(max_workers=2) as executor:
                    list(executor.map(write, records))
                # Both writer connections have closed; a fresh connection retrieves vectors.
                with backend.memory_store() as store:
                    self.assertEqual(len(store.search(namespace)), 2)
                    found = store.search(namespace, query="apple", limit=1)
                    self.assertEqual(found[0].key, "apple")
                    self.assertGreater(found[0].score, 0.99)
                    self.assertEqual(store.search((*namespace, "private"), query="apple"), [])
            finally:
                with backend.memory_store() as store:
                    for key, _ in records:
                        store.delete(namespace, key)


TEST_OWNER_URLS = {
    owner: f"TEST_{'' if owner == 'shared' else owner.upper() + '_'}MEMORY_DATABASE_URL"
    for owner in backend.MEMORY_OWNERS
}


@unittest.skipUnless(
    all(os.getenv(key) for key in TEST_OWNER_URLS.values()),
    "Set TEST_MEMORY_DATABASE_URL and TEST_{FRUIT,WEATHER,MITHRIDAT}_MEMORY_DATABASE_URL for isolated database tests",
)
class SeparatePostgresMemoryTests(unittest.TestCase):
    def test_independent_stores_parallel_writes_and_shared_recall(self):
        environment = {
            key.removeprefix("TEST_"): os.environ[key]
            for key in TEST_OWNER_URLS.values()
        }
        project = f"integration-{uuid4().hex}"
        shared_namespace = ("memories", project, "episodes", "shared")
        marker_namespace = ("integration_test", project)
        with patch.dict(os.environ, environment), patch.object(
            backend, "OllamaEmbeddings", return_value=TestEmbeddings()
        ), patch.object(episodic, "MEMORY_PROJECT", project), patch.object(
            episodic, "EPISODIC_NAMESPACE", shared_namespace
        ):
            owners = [None, *backend.SPECIALIST_PORTS]
            for owner in owners:
                with backend.memory_store(specialist=owner, initialize=True):
                    pass
            try:
                def write(owner):
                    with backend.memory_store(specialist=owner) as store:
                        # Identical namespace/key in every DB proves physical routing.
                        store.put(marker_namespace, "same-key", {"owner": owner}, index=False)
                        store.put(episodic.episode_namespace(owner), "episode", {"lesson": f"apple {owner or 'shared'} lesson"})

                with ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(write, owners))
                for owner in owners:
                    with backend.memory_store(specialist=owner) as store:
                        self.assertEqual(store.get(marker_namespace, "same-key").value, {"owner": owner})
                        for other in owners:
                            if other != owner:
                                self.assertIsNone(store.get(episodic.episode_namespace(other), "episode"))
                for owner in backend.SPECIALIST_PORTS:
                    context = episodic.recall_context("apple", specialist=owner)
                    self.assertIn(f"apple {owner} lesson", context)
                    self.assertIn("apple shared lesson", context)
                    for other in backend.SPECIALIST_PORTS:
                        if other != owner:
                            self.assertNotIn(f"apple {other} lesson", context)
                shared_context = episodic.recall_context("apple")
                self.assertIn("apple shared lesson", shared_context)
                self.assertNotIn("apple fruit lesson", shared_context)

                connect = backend.memory_store

                def unavailable_shared(*, specialist=None):
                    if specialist is None:
                        raise ConnectionError("shared unavailable")
                    return connect(specialist=specialist)

                with patch.object(episodic, "memory_store", side_effect=unavailable_shared), self.assertLogs(episodic.logger, "WARNING"):
                    context = episodic.recall_context("apple", specialist="fruit")
                    self.assertIn("apple fruit lesson", context)
            finally:
                for owner in owners:
                    with backend.memory_store(specialist=owner) as store:
                        store.delete(marker_namespace, "same-key")
                        store.delete(episodic.episode_namespace(owner), "episode")


if __name__ == "__main__":
    unittest.main()
