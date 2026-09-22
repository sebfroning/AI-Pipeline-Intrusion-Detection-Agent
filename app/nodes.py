from app.state import AgentState
from langgraph.types import Command, Send
from typing import Literal

def supervisor(state: AgentState) -> Command[Literal["detector", "auditor", "cleaner", "reporter", "__end__"]]:
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

    """ planned triggers:
    "triggers": [
        {"on": "detect.backdoored", "do": "invoke_cleaner"},
        {"on": "audit.backdoored", "do": "invoke_cleaner"},
        {"on": "clean.succeeded", "do": "invoke_detector"},
    ]
    """

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

def detector(state: AgentState) -> dict:
    return {
        "jobs": [{
            "job_id": state["current_job_id"],
            "agent": "detector",
            "status": "completed",
            "artifact": {},
            "event": "detect.backdoored", ## list: ["detect.backdoored", "detect.success", "detect.uncertain", "detect.failed"]
            "result": {},
        }]
    }

def auditor(state: AgentState) -> dict:
    return {
        "jobs": [{
            "job_id": state["current_job_id"],
            "agent": "auditor",
            "status": "completed",
            "artifact": {},
            "event": "", ## list: ["audit.backdoored", "audit.success", "audit.suspicious", "audit.failed"]
            "result": {},
        }]
    }

def cleaner(state: AgentState) -> dict:
    return {
        "jobs": [{
            "job_id": state["current_job_id"],
            "agent": "cleaner",
            "status": "completed",
            "artifact": {},
            "event": "", ## list: ["clean.success", "clean.failed"]
            "result": {},
        }]
    }

def reporter(state: AgentState) -> dict:
    return {}