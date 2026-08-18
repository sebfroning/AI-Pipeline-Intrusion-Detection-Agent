import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain.messages import AIMessage

from app.agents.supervisor import (
    ask_fruit_specialist,
    ask_weather_specialist,
    create_supervisor,
)
from app.state import OversightState


class SupervisorTests(unittest.TestCase):
    @patch("app.agents.supervisor.create_agent")
    def test_sequential_mode_limits_each_response_to_one_call(self, create) -> None:
        create_supervisor(parallel_specialists=False)

        prompt = create.call_args.kwargs["system_prompt"]
        self.assertIn(
            "Never request more than one specialist tool call in a single response",
            prompt,
        )

    @patch("app.agents.supervisor.create_agent")
    def test_parallel_mode_requests_both_calls_in_one_response(self, create) -> None:
        create_supervisor(parallel_specialists=True)

        prompt = create.call_args.kwargs["system_prompt"]
        self.assertIn("request exactly one call to each specialist", prompt)
        self.assertIn("in the same response", prompt)

    @patch("app.agents.supervisor.create_agent")
    def test_supervisor_uses_shared_oversight_state(self, create: Mock) -> None:
        create_supervisor()

        self.assertIs(create.call_args.kwargs["state_schema"], OversightState)

    @patch("app.agents.supervisor.fruit_specialist")
    def test_fruit_specialist_inherits_oversight_metadata(self, specialist: Mock) -> None:
        specialist.invoke.return_value = {"messages": [AIMessage(content="done")]}
        metadata = {
            "target_name": "classifier",
            "target_kind": "model",
            "access_mode": "black_box",
            "available_interfaces": ["inference API"],
        }

        ask_fruit_specialist.func(
            question="inspect apples",
            runtime=SimpleNamespace(state={"oversight": metadata}),
        )

        self.assertIs(specialist.invoke.call_args.args[0]["oversight"], metadata)

    @patch("app.agents.supervisor.weather_specialist")
    def test_weather_specialist_inherits_oversight_metadata(
        self, specialist: Mock
    ) -> None:
        specialist.invoke.return_value = {"messages": [AIMessage(content="done")]}
        metadata = {
            "target_name": "classifier",
            "target_kind": "model",
            "access_mode": "white_box",
            "available_interfaces": ["weights"],
        }

        ask_weather_specialist.func(
            question="inspect weather",
            runtime=SimpleNamespace(state={"oversight": metadata}),
        )

        self.assertIs(specialist.invoke.call_args.args[0]["oversight"], metadata)


if __name__ == "__main__":
    unittest.main()
