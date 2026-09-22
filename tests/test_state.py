import unittest

from langchain.agents.middleware import ModelRequest
from langchain.messages import AIMessage

from app.config import model
from app.state import format_oversight_metadata, include_oversight_metadata


class OversightStateTests(unittest.TestCase):
    def test_formats_optional_metadata_for_agent_prompts(self) -> None:
        rendered = format_oversight_metadata(
            {
                "target_name": "fraud classifier",
                "target_kind": "pipeline",
                "access_mode": "gray_box",
                "description": "Scores transactions",
                "available_interfaces": ["inference API", "request logs"],
            }
        )

        self.assertIn("Access mode: gray_box", rendered)
        self.assertIn("Description: Scores transactions", rendered)
        self.assertIn("inference API, request logs", rendered)

    def test_middleware_adds_metadata_to_existing_system_prompt(self) -> None:
        request = ModelRequest(
            model=model,
            messages=[],
            system_prompt="You are a specialist.",
            state={
                "messages": [],
                "oversight": {
                    "target_name": "fraud classifier",
                    "target_kind": "model",
                    "access_mode": "black_box",
                },
            },
        )
        captured = []

        def handler(updated_request: ModelRequest) -> AIMessage:
            captured.append(updated_request.system_prompt)
            return AIMessage(content="done")

        include_oversight_metadata.wrap_model_call(request, handler)

        self.assertIn("You are a specialist.", captured[0])
        self.assertIn("Target name: fraud classifier", captured[0])
        self.assertIn("Access mode: black_box", captured[0])
        self.assertIn("Available interfaces: none declared", captured[0])

    def test_middleware_adds_recalled_episodes_to_system_prompt(self) -> None:
        request = ModelRequest(
            model=model,
            messages=[],
            system_prompt="You are a specialist.",
            state={
                "messages": [],
                "oversight": {
                    "target_name": "classifier",
                    "target_kind": "model",
                    "access_mode": "white_box",
                },
                "episodic_context": "<relevant_episodes>prior run</relevant_episodes>",
            },
        )
        captured = []

        def handler(updated_request: ModelRequest) -> AIMessage:
            captured.append(updated_request.system_prompt)
            return AIMessage(content="done")

        include_oversight_metadata.wrap_model_call(request, handler)

        self.assertIn("<relevant_episodes>prior run</relevant_episodes>", captured[0])


if __name__ == "__main__":
    unittest.main()
