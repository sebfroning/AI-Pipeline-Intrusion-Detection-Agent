import json

from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.audit_specialist import audit_specialist
from app.agents.clean_specialist import clean_specialist
from app.agents.detect_specialist import detect_specialist
from app.config import model
from app.state import OversightState, SpecialistResult, include_oversight_metadata


def _invoke_specialist(
    specialist,
    name: str,
    tool_name: str,
    question: str,
    runtime: ToolRuntime[None, OversightState],
) -> Command:
    result = specialist.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "oversight": runtime.state["oversight"],
        }
    )
    finding = result["messages"][-1].text
    specialist_result: SpecialistResult = {
        "specialist": name,
        "question": question,
        "finding": finding,
    }
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=finding,
                    tool_call_id=runtime.tool_call_id,
                    name=tool_name,
                )
            ],
            "specialist_results": [specialist_result],
        }
    )


@tool
def invoke_detect_specialist(
    question: str, runtime: ToolRuntime[None, OversightState]
) -> Command:
    """Invoke the detect specialist to use inference-time detection to find backdoors."""
    return _invoke_specialist(
        detect_specialist,
        "detect",
        "invoke_detect_specialist",
        question,
        runtime,
    )


@tool
def invoke_clean_specialist(
    question: str, runtime: ToolRuntime[None, OversightState]
) -> Command:
    """Invoke the clean specialist to clean the potentially backdoored model."""
    return _invoke_specialist(
        clean_specialist,
        "clean",
        "invoke_clean_specialist",
        question,
        runtime,
    )


@tool
def invoke_audit_specialist(
    question: str, runtime: ToolRuntime[None, OversightState]
) -> Command:
    """Invoke the audit specialist to audit the model's state."""
    return _invoke_specialist(
        audit_specialist,
        "audit",
        "invoke_audit_specialist",
        question,
        runtime,
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
            "When multiple specialists are relevant, request those tool calls "
            "in the same response so they can run in parallel. "
        )
    else:
        dispatch_instructions = (
            "Never request more than one specialist tool call in a single response. "
        )

    supervisor_agent = create_agent(
        model=model,
        tools=[
            invoke_detect_specialist,
            invoke_clean_specialist,
            invoke_audit_specialist,
        ],
        state_schema=OversightState,
        middleware=[include_oversight_metadata],
        system_prompt=(
            "You are a supervisor with the task of detecting and cleaning "
            "backdoors in a pretrained image model. Use the tools available to "
            "you to detect and clean the backdoors. The available tools are: "
            "invoke_detect_specialist, invoke_clean_specialist, and "
            "invoke_audit_specialist. Plan which specialists to invoke and in "
            "what order to ensure the model is clean and any backdoors are "
            "removed. Pass each specialist a focused sub-question. "
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
