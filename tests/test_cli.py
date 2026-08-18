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
            {"messages": [{"role": "user", "content": "hello"}]},
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
            {"messages": [{"role": "user", "content": "hello"}]},
            config={"max_concurrency": 2},
        )


if __name__ == "__main__":
    unittest.main()
