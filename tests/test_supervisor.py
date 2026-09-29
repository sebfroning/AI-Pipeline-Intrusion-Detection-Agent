import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from langchain.messages import AIMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.runnables import RunnableLambda

from app.agents.supervisor import (
    ask_fruit_specialist,
    ask_weather_specialist,
    create_supervisor,
    finalize_report,
    learn_from_run,
    recall_episodes,
)
from app.state import OversightState


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Fake model that supports the tool binding required by create_agent."""

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> "ToolCallingFakeModel":
        return self


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        recall = patch("app.agents.supervisor.recall_context", return_value="")
        recall.start()
        self.addCleanup(recall.stop)

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
    def test_fruit_specialist_inherits_oversight_metadata(
        self, specialist: Mock
    ) -> None:
        specialist.invoke.return_value = {"messages": [AIMessage(content="done")]}
        metadata = {
            "target_name": "classifier",
            "target_kind": "model",
            "access_mode": "black_box",
            "available_interfaces": ["inference API"],
        }

        command = ask_fruit_specialist.func(
            question="inspect apples",
            runtime=SimpleNamespace(
                state={"oversight": metadata}, tool_call_id="fruit-call"
            ),
        )

        self.assertIs(specialist.invoke.call_args.args[0]["oversight"], metadata)
        self.assertEqual(
            command.update["specialist_results"],
            [
                {
                    "specialist": "fruit",
                    "question": "inspect apples",
                    "finding": "done",
                }
            ],
        )
        self.assertEqual(command.update["messages"][0].tool_call_id, "fruit-call")

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

        command = ask_weather_specialist.func(
            question="inspect weather",
            runtime=SimpleNamespace(
                state={"oversight": metadata}, tool_call_id="weather-call"
            ),
        )

        self.assertIs(specialist.invoke.call_args.args[0]["oversight"], metadata)
        self.assertEqual(
            command.update["specialist_results"],
            [
                {
                    "specialist": "weather",
                    "question": "inspect weather",
                    "finding": "done",
                }
            ],
        )
        self.assertEqual(command.update["messages"][0].tool_call_id, "weather-call")

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
                    "specialist": "fruit",
                    "question": "fruit question",
                    "finding": "fruit finding",
                },
                {
                    "specialist": "weather",
                    "question": "weather question",
                    "finding": "weather finding",
                },
            ],
            "episodic_context": "past run guidance",
        }

        update = finalize_report(state)

        finalizer_input = finalizer_model.invoke.call_args.args[0][1].text
        self.assertIn('"specialist": "fruit"', finalizer_input)
        self.assertIn('"finding": "fruit finding"', finalizer_input)
        self.assertIn('"specialist": "weather"', finalizer_input)
        self.assertIn('"finding": "weather finding"', finalizer_input)
        self.assertIn("past run guidance", finalizer_input)
        self.assertEqual(update["final_report"], "combined report")

    @patch("app.agents.supervisor.model")
    @patch("app.agents.supervisor.create_agent")
    @patch("app.agents.supervisor.extract_episode")
    def test_finalizer_always_runs_after_supervisor(
        self,
        _extract: Mock,
        create: Mock,
        finalizer_model: Mock,
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

    @patch("app.agents.supervisor.extract_episode")
    @patch("app.agents.supervisor.fruit_specialist")
    def test_specialist_result_flows_through_supervisor_to_finalizer(
        self, specialist: Mock, _extract: Mock
    ) -> None:
        fake_model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_fruit_specialist",
                            "args": {"question": "inspect apples"},
                            "id": "fruit-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="supervisor synthesis"),
                AIMessage(content="combined final report"),
            ]
        )
        specialist.invoke.return_value = {
            "messages": [AIMessage(content="structured fruit finding")]
        }

        with patch("app.agents.supervisor.model", fake_model):
            result = create_supervisor().invoke(
                {
                    "messages": [{"role": "user", "content": "inspect apples"}],
                    "oversight": {
                        "target_name": "fruit model",
                        "target_kind": "model",
                        "access_mode": "white_box",
                    },
                }
            )

        self.assertEqual(
            result["specialist_results"],
            [
                {
                    "specialist": "fruit",
                    "question": "inspect apples",
                    "finding": "structured fruit finding",
                }
            ],
        )
        self.assertEqual(result["final_report"], "combined final report")

    @patch("app.agents.supervisor.recall_context", return_value="past episode")
    def test_recall_node_retrieves_context(
        self, recall: Mock
    ) -> None:
        state = {
            "messages": [{"role": "user", "content": "inspect this model"}],
            "oversight": {
                "target_name": "classifier",
                "target_kind": "model",
                "access_mode": "white_box",
            },
        }

        update = recall_episodes(state)

        self.assertIn("inspect this model", recall.call_args.args[0])
        self.assertEqual(update, {"episodic_context": "past episode"})

    @patch("app.agents.supervisor.extract_episode")
    def test_learning_node_extracts_shared_and_specialist_experiences(
        self, extract: Mock
    ) -> None:
        state = {
            "messages": [{"role": "user", "content": "inspect this model"}],
            "oversight": {
                "target_name": "classifier",
                "target_kind": "model",
                "access_mode": "white_box",
            },
            "specialist_results": [
                {
                    "specialist": "fruit",
                    "question": "inspect",
                    "finding": "finding",
                }
            ],
            "final_report": "completed report",
        }

        update = learn_from_run(state)

        trajectory = extract.call_args_list[0].args[0]
        self.assertEqual(trajectory["user_request"], "inspect this model")
        self.assertEqual(trajectory["final_report"], "completed report")
        self.assertEqual(
            trajectory["specialist_results"], state["specialist_results"]
        )
        self.assertEqual(extract.call_count, 2)
        specialist_call = extract.call_args_list[1]
        self.assertEqual(specialist_call.kwargs["specialist"], "fruit")
        self.assertEqual(specialist_call.args[0]["specialist_result"], state["specialist_results"][0])
        self.assertNotIn("final_report", specialist_call.args[0])
        self.assertEqual(update, {})

    @patch("app.agents.supervisor.recall_context", side_effect=RuntimeError("database offline"))
    def test_recall_failure_does_not_abort_run(self, _recall):
        with self.assertLogs("app.agents.supervisor", level="WARNING"):
            result = recall_episodes({"messages": [], "oversight": {}})
        self.assertEqual(result, {"episodic_context": ""})

    @patch("app.agents.supervisor.extract_episode", side_effect=[RuntimeError("shared failed"), None, None])
    def test_shared_failure_does_not_prevent_specialist_learning(self, extract):
        state = {
            "messages": [], "oversight": {}, "final_report": "report",
            "specialist_results": [
                {"specialist": "fruit", "question": "apples", "finding": "fruit result"},
                {"specialist": "weather", "question": "rain", "finding": "weather result"},
            ],
        }
        with self.assertLogs("app.agents.supervisor", level="WARNING"):
            self.assertEqual(learn_from_run(state), {})
        self.assertEqual(extract.call_count, 3)
        self.assertEqual(extract.call_args_list[2].kwargs["specialist"], "weather")
        self.assertEqual(state["final_report"], "report")

    @patch("app.agents.supervisor.recall_context", return_value="fruit history")
    @patch("app.agents.supervisor.fruit_specialist")
    def test_specialist_retrieves_focused_memory_and_retains_tool_observations(self, specialist, recall):
        specialist.invoke.return_value = {"messages": [
            ToolMessage(content="tool evidence", tool_call_id="lookup", name="fruit_info"),
            AIMessage(content="finding"),
        ]}
        result = ask_fruit_specialist.func(
            question="inspect apples",
            runtime=SimpleNamespace(state={"oversight": {}}, tool_call_id="fruit-call"),
        )
        self.assertEqual(recall.call_args.kwargs["specialist"], "fruit")
        self.assertIn("inspect apples", recall.call_args.args[0])
        self.assertEqual(specialist.invoke.call_args.args[0]["episodic_context"], "fruit history")
        self.assertEqual(result.update["specialist_results"][0]["tool_observations"], [
            {"tool": "fruit_info", "content": "tool evidence"}
        ])


if __name__ == "__main__":
    unittest.main()
