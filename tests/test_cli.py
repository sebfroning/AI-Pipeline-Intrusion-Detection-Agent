import sys
import unittest
from unittest.mock import Mock, patch

from langchain.messages import AIMessage

from app.__main__ import build_parser, main


class CliTests(unittest.TestCase):
    def test_parallel_specialists_flag_defaults_to_false(self) -> None:
        args = build_parser().parse_args(["hello"])

        self.assertFalse(args.parallel_specialists)

    def test_parallel_specialists_flag_enables_two_concurrent_calls(self) -> None:
        args = build_parser().parse_args(["--parallel-specialists", "hello"])

        self.assertTrue(args.parallel_specialists)

    @patch("app.__main__.create_supervisor")
    def test_main_preserves_sequential_execution_by_default(self, create: Mock) -> None:
        graph = create.return_value
        graph.invoke.return_value = {"messages": [AIMessage(content="done")]}

        with patch.object(sys, "argv", ["python -m app", "hello"]), patch(
            "builtins.print"
        ):
            main()

        create.assert_called_once_with(parallel_specialists=False)
        graph.invoke.assert_called_once_with(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "oversight": {
                    "target_name": "local model",
                    "target_kind": "model",
                    "access_mode": "black_box",
                    "available_interfaces": [],
                },
            },
            config={"max_concurrency": 1},
        )

    @patch("app.__main__.create_supervisor")
    def test_main_passes_parallel_mode_and_concurrency(self, create: Mock) -> None:
        graph = create.return_value
        graph.invoke.return_value = {"messages": [AIMessage(content="done")]}

        with patch.object(
            sys,
            "argv",
            ["python -m app", "--parallel-specialists", "hello"],
        ), patch("builtins.print"):
            main()

        create.assert_called_once_with(parallel_specialists=True)
        graph.invoke.assert_called_once_with(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "oversight": {
                    "target_name": "local model",
                    "target_kind": "model",
                    "access_mode": "black_box",
                    "available_interfaces": [],
                },
            },
            config={"max_concurrency": 2},
        )

    @patch("app.__main__.create_supervisor")
    def test_main_stores_cli_oversight_metadata_in_state(self, create: Mock) -> None:
        graph = create.return_value
        graph.invoke.return_value = {"messages": [AIMessage(content="done")]}

        argv = [
            "python -m app",
            "--target-name",
            "fraud classifier",
            "--target-kind",
            "pipeline",
            "--access-mode",
            "white_box",
            "--target-description",
            "Screens incoming transactions",
            "--available-interface",
            "weights",
            "--available-interface",
            "activations",
            "inspect it",
        ]
        with patch.object(sys, "argv", argv), patch("builtins.print"):
            main()

        state = graph.invoke.call_args.args[0]
        self.assertEqual(
            state["oversight"],
            {
                "target_name": "fraud classifier",
                "target_kind": "pipeline",
                "access_mode": "white_box",
                "description": "Screens incoming transactions",
                "available_interfaces": ["weights", "activations"],
            },
        )


if __name__ == "__main__":
    unittest.main()
