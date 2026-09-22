from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
from langgraph.types import Command, Send
from langgraph.graph import END
import operator

class Job(TypedDict):
    job_id: int
    agent: Literal["detector", "auditor", "cleaner", "reporter"]
    status: Literal["pending", "running", "completed", "failed"]
    artifact: dict
    event: str
    result: dict

class AgentState(TypedDict):
    plan: dict
    hops: int
    cleans: int
    artifact: dict
    next_job_id: int
    current_job_id: int
    jobs: Annotated[list[Job], operator.add]
    num_prev_specialists: int # how many specialists were invoked in the last call

def supervisor(state: AgentState) -> Command[Literal["detector", "auditor", "cleaner", "reporter", "__end__"]]:
    plan = state["plan"]
    jobs = state.get("jobs") or []
    next_id = state.get("next_job_id") or 1
    
    # If enough has been done, return to reporter
    if state["hops"] >= plan["budgets"]["max_hops"]:
        return Command(goto="reporter")

    # First, we visit the detector and auditor in parallel, as set by plan["initial_calls"]
    if not jobs:
        calls = plan["initial_calls"]
        return Command(
            goto=[Send(call["agent"], {**state, "current_job_id": next_id + i}) for i, call in enumerate(calls)],
            update={
                "hops": state["hops"] + 1,
                "next_job_id": next_id + len(calls),
                "num_prev_specialists": len(calls),
            },
        )

    # Next, look at the latest job(s) and see what to do next
    prev_jobs = jobs[-state["num_prev_specialists"]:]

    events = [job["event"] for job in prev_jobs]
    triggers = plan["triggers"]

    for trigger in triggers:
        if trigger["on"] in events:
            # If the max number of cleans has been reached, return to reporter
            if trigger["do"] == "invoke_cleaner" and state["cleans"] >= plan["budgets"]["max_cleans"]:
                return Command(goto="reporter")
            # Otherwise, invoke the cleaner or detector
            agent_name = trigger["do"].removeprefix("invoke_")  # "cleaner" / "detector"
            return Command(goto=agent_name, 
            update = {
                "hops": state["hops"] + 1,
                "current_job_id": next_id,
                "next_job_id": next_id + 1,
                "cleans": state["cleans"] + 1 if agent_name == "cleaner" else state["cleans"],
                "num_prev_specialists": 1,
            })

    return Command(goto="reporter")


def detector(state: AgentState) -> dict:
    return

def auditor(state: AgentState) -> dict:
    return

def cleaner(state: AgentState) -> dict:
    return

def reporter(state: AgentState) -> dict:
    return