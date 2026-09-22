"""Shared state for the supervisor hierarchy and the specialist graph."""

import operator
from typing import Annotated, Literal, NotRequired, Required, TypedDict, cast

from langchain.agents import AgentState as LangChainAgentState
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langgraph.types import Command, Send


TargetKind = Literal["model", "agent", "pipeline"]
AccessMode = Literal["black_box", "gray_box", "white_box"]
SpecialistName = Literal["detect", "clean", "audit", "fruit", "weather"]
PipelineAgent = Literal["detector", "auditor", "cleaner", "reporter"]


class OversightMetadata(TypedDict):
    """Facts about the system being overseen.

    ``available_interfaces`` names concrete sources a specialist can use, such as
    an inference API, traces, activations, weights, or training data.
    """

    target_name: Required[str]
    target_kind: Required[TargetKind]
    access_mode: Required[AccessMode]
    description: NotRequired[str]
    available_interfaces: NotRequired[list[str]]


class SpecialistResult(TypedDict):
    """A structured finding produced by one specialist invocation."""

    specialist: Required[SpecialistName]
    question: Required[str]
    finding: Required[str]


class OversightState(LangChainAgentState):
    """Agent state shared across the complete supervisor hierarchy."""

    oversight: Required[OversightMetadata]
    specialist_results: NotRequired[
        Annotated[list[SpecialistResult], operator.add]
    ]
    final_report: NotRequired[str]


class Job(TypedDict):
    job_id: int
    agent: PipelineAgent
    status: Literal["pending", "running", "completed", "failed"]
    artifact: dict
    event: str
    result: dict


class AgentState(TypedDict):
    """Graph state for the detect / audit / clean / report pipeline."""

    plan: dict
    hops: int
    cleans: int
    artifact: dict
    next_job_id: int
    current_job_id: int
    jobs: Annotated[list[Job], operator.add]
    num_prev_specialists: int
    oversight: NotRequired[OversightMetadata]


def format_oversight_metadata(metadata: OversightMetadata) -> str:
    """Render oversight metadata for a specialist's system prompt."""
    lines = [
        f"- Target name: {metadata['target_name']}",
        f"- Target kind: {metadata['target_kind']}",
        f"- Access mode: {metadata['access_mode']}",
    ]
    if description := metadata.get("description"):
        lines.append(f"- Description: {description}")
    interfaces = metadata.get("available_interfaces", [])
    lines.append(
        "- Available interfaces: "
        + (", ".join(interfaces) if interfaces else "none declared")
    )
    return "\n".join(lines)


@dynamic_prompt
def include_oversight_metadata(request: ModelRequest) -> str:
    """Add the shared oversight context to an agent's own system prompt."""
    base_prompt = request.system_prompt or ""
    metadata = cast(OversightState, request.state)["oversight"]
    return (
        base_prompt
        + "\n\nChoose tools that are compatible with the declared access mode and "
        "available interfaces.\n\nOversight metadata:\n"
        + format_oversight_metadata(metadata)
    )


def supervisor(
    state: AgentState,
) -> Command[Literal["detector", "auditor", "cleaner", "reporter", "__end__"]]:
    """Route the next specialist job from the current plan and job history."""
    plan = state["plan"]
    jobs = state.get("jobs") or []
    next_id = state.get("next_job_id") or 1

    if state["hops"] >= plan["budgets"]["max_hops"]:
        return Command(goto="reporter")

    if not jobs:
        calls = plan["initial_calls"]
        return Command(
            goto=[
                Send(call["agent"], {**state, "current_job_id": next_id + i})
                for i, call in enumerate(calls)
            ],
            update={
                "hops": state["hops"] + 1,
                "next_job_id": next_id + len(calls),
                "num_prev_specialists": len(calls),
            },
        )

    prev_jobs = jobs[-state["num_prev_specialists"] :]
    events = [job["event"] for job in prev_jobs]
    triggers = plan["triggers"]

    for trigger in triggers:
        if trigger["on"] in events:
            if (
                trigger["do"] == "invoke_cleaner"
                and state["cleans"] >= plan["budgets"]["max_cleans"]
            ):
                return Command(goto="reporter")
            agent_name = trigger["do"].removeprefix("invoke_")
            return Command(
                goto=agent_name,
                update={
                    "hops": state["hops"] + 1,
                    "current_job_id": next_id,
                    "next_job_id": next_id + 1,
                    "cleans": (
                        state["cleans"] + 1
                        if agent_name == "cleaner"
                        else state["cleans"]
                    ),
                    "num_prev_specialists": 1,
                },
            )

    return Command(goto="reporter")
