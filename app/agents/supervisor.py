import json
import logging
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.fruit_specialist import fruit_specialist
from app.agents.weather_specialist import weather_specialist
from app.config import model
from app.memory import extract_episode, recall_context
from app.state import (
    OversightState,
    SpecialistResult,
    include_oversight_metadata,
)


logger = logging.getLogger(__name__)


def _first_user_request(state: OversightState) -> str:
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            return message.text
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def recall_episodes(state: OversightState) -> dict:
    """Retrieve shared past runs relevant to the current request."""
    query = json.dumps(
        {
            "request": _first_user_request(state),
            "oversight": state["oversight"],
        },
        default=str,
    )
    try:
        context = recall_context(query)
    except Exception as exc:
        logger.warning(
            "Episodic memory recall failed; continuing without it: %s", exc
        )
        context = ""
    return {"episodic_context": context}


def _specialist_context(name: str, question: str, state: OversightState) -> str:
    query = json.dumps({"question": question, "oversight": state["oversight"]})
    try:
        return recall_context(query, specialist=name)
    except Exception as exc:
        logger.warning("%s memory recall failed; continuing without it: %s", name, exc)
        return ""


def _tool_observations(result: dict) -> list[dict]:
    """Preserve observable tool outputs, without model reasoning messages."""
    return [
        {"tool": message.name, "content": message.text}
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]


@tool
def ask_fruit_specialist(
    question: str, runtime: ToolRuntime[None, OversightState]
) -> Command:
    """Ask the fruit specialist a focused fruit question."""
    result = fruit_specialist.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "oversight": runtime.state["oversight"],
            "episodic_context": _specialist_context("fruit", question, runtime.state),
        }
    )
    finding = result["messages"][-1].text
    specialist_result: SpecialistResult = {
        "specialist": "fruit",
        "question": question,
        "finding": finding,
    }
    if observations := _tool_observations(result):
        specialist_result["tool_observations"] = observations
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=finding,
                    tool_call_id=runtime.tool_call_id,
                    name="ask_fruit_specialist",
                )
            ],
            "specialist_results": [specialist_result],
        }
    )


@tool
def ask_weather_specialist(
    question: str, runtime: ToolRuntime[None, OversightState]
) -> Command:
    """Ask the weather specialist for a harvest weather pattern."""
    result = weather_specialist.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "oversight": runtime.state["oversight"],
            "episodic_context": _specialist_context("weather", question, runtime.state),
        }
    )
    finding = result["messages"][-1].text
    specialist_result: SpecialistResult = {
        "specialist": "weather",
        "question": question,
        "finding": finding,
    }
    if observations := _tool_observations(result):
        specialist_result["tool_observations"] = observations
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=finding,
                    tool_call_id=runtime.tool_call_id,
                    name="ask_weather_specialist",
                )
            ],
            "specialist_results": [specialist_result],
        }
    )


FINALIZER_PROMPT = (
    "You are the finalizer for an oversight workflow. Produce one clear final "
    "report for the user. Use every structured specialist result, reconcile "
    "overlapping or conflicting findings, and preserve important uncertainty. "
    "The supervisor synthesis is supporting context, not a substitute for the "
    "structured findings. If no specialist was used, refine the supervisor's "
    "answer. Treat past episodes only as guidance about approach, never as "
    "evidence for the current target's verdict. Do not discuss internal routing "
    "or the finalization process."
)


def finalize_report(state: OversightState) -> dict:
    """Combine structured specialist findings into the workflow's final report."""
    messages = state["messages"]
    user_request = next(
        (message.text for message in messages if isinstance(message, HumanMessage)),
        "",
    )
    supervisor_synthesis = messages[-1].text if messages else ""
    specialist_results = state.get("specialist_results", [])
    finalizer_input = {
        "user_request": user_request,
        "oversight_metadata": state["oversight"],
        "specialist_results": specialist_results,
        "supervisor_synthesis": supervisor_synthesis,
        "relevant_past_episodes": state.get("episodic_context", ""),
    }
    response = model.invoke(
        [
            SystemMessage(content=FINALIZER_PROMPT),
            HumanMessage(content=json.dumps(finalizer_input, indent=2)),
        ]
    )
    return {
        "messages": [response],
        "final_report": response.text,
    }


def learn_from_run(state: OversightState) -> dict:
    """Extract and persist reusable experience from the completed run."""
    trajectory = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_request": _first_user_request(state),
        "oversight_metadata": state["oversight"],
        "specialist_results": state.get("specialist_results", []),
        "final_report": state.get("final_report", ""),
    }
    try:
        extract_episode(trajectory, model=model)
    except Exception as exc:
        logger.warning(
            "Episodic memory extraction failed; report is unaffected: %s", exc
        )
    for finding in state.get("specialist_results", []):
        try:
            extract_episode(
                {
                    "timestamp": trajectory["timestamp"],
                    "oversight_metadata": state["oversight"],
                    "specialist_result": finding,
                },
                model=model,
                specialist=finding["specialist"],
            )
        except Exception as exc:
            logger.warning(
                "%s memory extraction failed; report is unaffected: %s",
                finding["specialist"],
                exc,
            )
    return {}


def create_supervisor(*, parallel_specialists: bool = False):
    """Create the supervisor followed by an unconditional finalizer node."""
    if parallel_specialists:
        dispatch_instructions = (
            "When both specialists are relevant, request exactly one call to each "
            "specialist in the same response so they can run in parallel. "
        )
    else:
        dispatch_instructions = (
            "Never request more than one specialist tool call in a single response. "
        )

    supervisor_agent = create_agent(
        model=model,
        tools=[ask_fruit_specialist, ask_weather_specialist],
        state_schema=OversightState,
        middleware=[include_oversight_metadata],
        system_prompt=(
            "You are a supervisor with a fruit specialist and a harvest-weather "
            "specialist. Answer unrelated prompts yourself without calling a tool. "
            "For fruit questions, call ask_fruit_specialist. For requests for a "
            "harvest weather pattern, call ask_weather_specialist. If both are "
            "relevant, call both and combine their results. Pass each specialist a "
            "focused sub-question. "
            + dispatch_instructions
            + "Never call the same specialist more than once per user "
            "prompt. After any tool calls, provide a concise synthesis for the "
            "finalizer."
        ),
    )

    workflow = StateGraph(OversightState)
    workflow.add_node("recall_episodes", recall_episodes)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("finalizer", finalize_report)
    workflow.add_node("learn_from_run", learn_from_run)
    workflow.add_edge(START, "recall_episodes")
    workflow.add_edge("recall_episodes", "supervisor")
    workflow.add_edge("supervisor", "finalizer")
    workflow.add_edge("finalizer", "learn_from_run")
    workflow.add_edge("learn_from_run", END)
    return workflow.compile()


# Preserve the original import for callers that want sequential execution.
supervisor = create_supervisor()
