import json

from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.fruit_specialist import fruit_specialist
from app.agents.weather_specialist import weather_specialist
from app.config import model
from app.state import (
    OversightState,
    SpecialistResult,
    include_oversight_metadata,
)


@tool
def ask_fruit_specialist(
    question: str, runtime: ToolRuntime[None, OversightState]
) -> Command:
    """Ask the fruit specialist a focused fruit question."""
    result = fruit_specialist.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "oversight": runtime.state["oversight"],
        }
    )
    finding = result["messages"][-1].text
    specialist_result: SpecialistResult = {
        "specialist": "fruit",
        "question": question,
        "finding": finding,
    }
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
        }
    )
    finding = result["messages"][-1].text
    specialist_result: SpecialistResult = {
        "specialist": "weather",
        "question": question,
        "finding": finding,
    }
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
    "answer. Do not discuss internal routing or the finalization process."
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
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("finalizer", finalize_report)
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "finalizer")
    workflow.add_edge("finalizer", END)
    return workflow.compile()


# Preserve the original import for callers that want sequential execution.
supervisor = create_supervisor()
