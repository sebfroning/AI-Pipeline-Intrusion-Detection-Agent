import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from langchain.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.runnables import RunnableLambda

from app.agents.supervisor import (
    create_supervisor,
    finalize_report,
    invoke_audit_specialist,
    invoke_detect_specialist,
)
from app.state import OversightState


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Fake model that supports the tool binding required by create_agent."""

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> "ToolCallingFakeModel":
        return self


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
    def test_parallel_mode_requests_calls_in_one_response(self, create) -> None:
        create_supervisor(parallel_specialists=True)

        prompt = create.call_args.kwargs["system_prompt"]
        self.assertIn("request those tool calls", prompt)
        self.assertIn("in the same response", prompt)

    @patch("app.agents.supervisor.create_agent")
    def test_supervisor_uses_shared_oversight_state(self, create: Mock) -> None:
        create_supervisor()

        self.assertIs(create.call_args.kwargs["state_schema"], OversightState)

    @patch("app.agents.supervisor.detect_specialist")
    def test_detect_specialist_inherits_oversight_metadata(
        self, specialist: Mock
    ) -> None:
        specialist.invoke.return_value = {"messages": [AIMessage(content="done")]}
        metadata = {
            "target_name": "classifier",
            "target_kind": "model",
            "access_mode": "black_box",
            "available_interfaces": ["inference API"],
        }

        command = invoke_detect_specialist.func(
            question="inspect backdoors",
            runtime=SimpleNamespace(
                state={"oversight": metadata}, tool_call_id="detect-call"
            ),
        )

        self.assertIs(specialist.invoke.call_args.args[0]["oversight"], metadata)
        self.assertEqual(
            command.update["specialist_results"],
            [
                {
                    "specialist": "detect",
                    "question": "inspect backdoors",
                    "finding": "done",
                }
            ],
        )
        self.assertEqual(command.update["messages"][0].tool_call_id, "detect-call")

    @patch("app.agents.supervisor.audit_specialist")
    def test_audit_specialist_inherits_oversight_metadata(
        self, specialist: Mock
    ) -> None:
        specialist.invoke.return_value = {"messages": [AIMessage(content="done")]}
        metadata = {
            "target_name": "classifier",
            "target_kind": "model",
            "access_mode": "white_box",
            "available_interfaces": ["weights"],
        }

        command = invoke_audit_specialist.func(
            question="audit the model",
            runtime=SimpleNamespace(
                state={"oversight": metadata}, tool_call_id="audit-call"
            ),
        )

        self.assertIs(specialist.invoke.call_args.args[0]["oversight"], metadata)
        self.assertEqual(
            command.update["specialist_results"],
            [
                {
                    "specialist": "audit",
                    "question": "audit the model",
                    "finding": "done",
                }
            ],
        )
        self.assertEqual(command.update["messages"][0].tool_call_id, "audit-call")

    @patch("app.agents.supervisor.model")
    def test_finalizer_receives_all_structured_results(
        self, finalizer_model: Mock
    ) -> None:
        finalizer_model.invoke.return_value = AIMessage(content="combined report")
        state = {
            "messages": [
                AIMessage(content="supervisor synthesis"),
            ],
            "oversight": {
                "target_name": "classifier",
                "target_kind": "model",
                "access_mode": "white_box",
            },
            "specialist_results": [
                {
                    "specialist": "detect",
                    "question": "detect question",
                    "finding": "detect finding",
                },
                {
                    "specialist": "audit",
                    "question": "audit question",
                    "finding": "audit finding",
                },
            ],
        }

        update = finalize_report(state)

        finalizer_input = finalizer_model.invoke.call_args.args[0][1].text
        self.assertIn('"specialist": "detect"', finalizer_input)
        self.assertIn('"finding": "detect finding"', finalizer_input)
        self.assertIn('"specialist": "audit"', finalizer_input)
        self.assertIn('"finding": "audit finding"', finalizer_input)
        self.assertEqual(update["final_report"], "combined report")

    @patch("app.agents.supervisor.model")
    @patch("app.agents.supervisor.create_agent")
    def test_finalizer_always_runs_after_supervisor(
        self, create: Mock, finalizer_model: Mock
    ) -> None:
        create.return_value = RunnableLambda(
            lambda _state: {"messages": [AIMessage(content="supervisor answer")]}
        )
        finalizer_model.invoke.return_value = AIMessage(content="final report")
        graph = create_supervisor()

        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "oversight": {
                    "target_name": "classifier",
                    "target_kind": "model",
                    "access_mode": "black_box",
                },
            }
        )

        finalizer_model.invoke.assert_called_once()
        self.assertEqual(result["final_report"], "final report")
        self.assertEqual(result["messages"][-1].text, "final report")

    @patch("app.agents.supervisor.detect_specialist")
    def test_specialist_result_flows_through_supervisor_to_finalizer(
        self, specialist: Mock
    ) -> None:
        fake_model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "invoke_detect_specialist",
                            "args": {"question": "inspect backdoors"},
                            "id": "detect-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="supervisor synthesis"),
                AIMessage(content="combined final report"),
            ]
        )
        specialist.invoke.return_value = {
            "messages": [AIMessage(content="structured detect finding")]
        }

        with patch("app.agents.supervisor.model", fake_model):
            result = create_supervisor().invoke(
                {
                    "messages": [{"role": "user", "content": "inspect backdoors"}],
                    "oversight": {
                        "target_name": "resnet fixture",
                        "target_kind": "model",
                        "access_mode": "white_box",
                    },
                }
            )

        self.assertEqual(
            result["specialist_results"],
            [
                {
                    "specialist": "detect",
                    "question": "inspect backdoors",
                    "finding": "structured detect finding",
                }
            ],
        )
        self.assertEqual(result["final_report"], "combined final report")


if __name__ == "__main__":
    unittest.main()
