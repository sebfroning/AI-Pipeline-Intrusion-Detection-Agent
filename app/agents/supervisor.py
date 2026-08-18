from langchain.agents import create_agent
from langchain.tools import tool

from app.agents.fruit_specialist import fruit_specialist
from app.agents.weather_specialist import weather_specialist
from app.config import model


@tool
def ask_fruit_specialist(question: str) -> str:
    """Ask the fruit specialist a focused fruit question."""
    result = fruit_specialist.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return result["messages"][-1].text


@tool
def ask_weather_specialist(question: str) -> str:
    """Ask the weather specialist for a harvest weather pattern."""
    result = weather_specialist.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return result["messages"][-1].text


def create_supervisor(*, parallel_specialists: bool = False):
    """Create a supervisor with sequential or parallel specialist dispatch."""
    if parallel_specialists:
        dispatch_instructions = (
            "When both specialists are relevant, request exactly one call to each "
            "specialist in the same response so they can run in parallel. "
        )
    else:
        dispatch_instructions = (
            "Never request more than one specialist tool call in a single response. "
        )

    return create_agent(
        model=model,
        tools=[ask_fruit_specialist, ask_weather_specialist],
        system_prompt=(
            "You are a supervisor with a fruit specialist and a harvest-weather "
            "specialist. Answer unrelated prompts yourself without calling a tool. "
            "For fruit questions, call ask_fruit_specialist. For requests for a "
            "harvest weather pattern, call ask_weather_specialist. If both are "
            "relevant, call both and combine their results. Pass each specialist a "
            "focused sub-question. "
            + dispatch_instructions
            + "Never call the same specialist more than once per user "
            "prompt. After any tool calls, give the user the final answer."
        ),
    )


# Preserve the original import for callers that want sequential execution.
supervisor = create_supervisor()
