import os
import unittest
from unittest.mock import Mock, patch

from langgraph.store.memory import InMemoryStore
from psycopg.conninfo import conninfo_to_dict

from app.memory import backend


class MemoryBackendTests(unittest.TestCase):
    def test_each_specialist_uses_its_own_connection_settings(self):
        for specialist, port in backend.SPECIALIST_PORTS.items():
            with self.subTest(specialist=specialist), patch.dict(os.environ, {
                "MEMORY_DATABASE_URL": "postgresql://shared:secret@shared/shared",
                f"{specialist.upper()}_POSTGRES_PASSWORD": "a b'c\\d@:/?#",
            }, clear=True):
                parameters = conninfo_to_dict(backend.connection_string(specialist))
                self.assertEqual(parameters["port"], port)
                self.assertEqual(parameters["dbname"], f"{specialist}_memory")
                self.assertEqual(parameters["user"], f"{specialist}_agent")
                self.assertEqual(parameters["password"], "a b'c\\d@:/?#")

    def test_specialist_uri_overrides_only_its_own_settings(self):
        with patch.dict(os.environ, {
            "FRUIT_MEMORY_DATABASE_URL": "postgresql://fruit:secret@remote/fruit?sslmode=require",
            "MEMORY_DATABASE_URL": "postgresql://shared:secret@shared/shared",
        }, clear=True):
            parameters = conninfo_to_dict(backend.connection_string("fruit"))
            self.assertEqual(parameters["host"], "remote")
            self.assertEqual(parameters["sslmode"], "require")
            self.assertEqual(conninfo_to_dict(backend.connection_string())["host"], "shared")

    def test_missing_private_credentials_never_fall_back_to_shared(self):
        with patch.dict(os.environ, {"MEMORY_DATABASE_URL": "postgresql://shared:secret@shared/shared"}, clear=True):
            with self.assertRaisesRegex(ValueError, "FRUIT_POSTGRES_PASSWORD"):
                backend.connection_string("fruit")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                backend.connection_string("typo")

    def test_password_special_characters_are_preserved(self):
        with patch.dict(os.environ, {"POSTGRES_PASSWORD": "a b'c\\d@:/?#"}, clear=True):
            parameters = conninfo_to_dict(backend.connection_string())
        self.assertEqual(parameters["password"], "a b'c\\d@:/?#")
        self.assertEqual(parameters["host"], "127.0.0.1")
        self.assertEqual(parameters["connect_timeout"], "5")

    def test_remote_uri_overrides_local_configuration(self):
        with patch.dict(os.environ, {
            "MEMORY_DATABASE_URL": "postgresql://remote:secret@db.example/test?sslmode=require"
        }, clear=True):
            parameters = conninfo_to_dict(backend.connection_string())
        self.assertEqual(parameters["host"], "db.example")
        self.assertEqual(parameters["sslmode"], "require")

    def test_missing_credentials_fail_without_connecting(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "POSTGRES_PASSWORD"):
                backend.connection_string()

    def test_embedding_configuration_prevents_mixing_models(self):
        store = InMemoryStore()
        with patch.object(backend, "EMBED_MODEL", "original"), patch.object(backend, "EMBED_DIMS", 768):
            backend._check_embedding_config(store, initialize=True)
            backend._check_embedding_config(store, initialize=False)
        with patch.object(backend, "EMBED_MODEL", "different"):
            with self.assertRaisesRegex(ValueError, "differ"):
                backend._check_embedding_config(store, initialize=False)

    def test_connections_close_when_operation_fails(self):
        with patch.object(backend.PostgresStore, "from_conn_string") as connect, patch.object(
            backend, "connection_string", return_value="dbname=test"
        ), patch.object(backend, "embedding_config", return_value=None), patch.object(
            backend, "_check_embedding_config"
        ):
            with self.assertRaises(RuntimeError):
                with backend.memory_store() as store:
                    raise RuntimeError("failed")
            store.setup.assert_not_called()
            connect.return_value.__exit__.assert_called_once()

    def test_initialization_runs_migrations_explicitly(self):
        with patch.object(backend.PostgresStore, "from_conn_string") as connect, patch.object(
            backend, "connection_string", return_value="dbname=test"
        ), patch.object(backend, "embedding_config", return_value=None), patch.object(
            backend, "_check_embedding_config"
        ) as check:
            with backend.memory_store(initialize=True) as store:
                store.setup.assert_called_once_with()
                check.assert_called_once_with(store, initialize=True)


if __name__ == "__main__":
    unittest.main()
