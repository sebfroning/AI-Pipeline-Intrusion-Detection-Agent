"""Shared state carried by the supervisor, specialists, and finalizer."""

import operator
from typing import Annotated, Literal, NotRequired, Required, TypedDict, cast

from langchain.agents import AgentState
from langchain.agents.middleware import ModelRequest, dynamic_prompt


TargetKind = Literal["model", "agent", "pipeline"]
AccessMode = Literal["black_box", "gray_box", "white_box"]
SpecialistName = Literal["fruit", "weather"]


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


class OversightState(AgentState):
    """Agent state shared across the complete supervisor hierarchy."""

    oversight: Required[OversightMetadata]
    specialist_results: NotRequired[
        Annotated[list[SpecialistResult], operator.add]
    ]
    final_report: NotRequired[str]


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
