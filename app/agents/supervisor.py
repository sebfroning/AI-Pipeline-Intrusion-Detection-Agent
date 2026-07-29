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


supervisor = create_agent(
    model=model,
    tools=[ask_fruit_specialist, ask_weather_specialist],
    system_prompt=(
        "You are a supervisor with a fruit specialist and a harvest-weather "
        "specialist. Answer unrelated prompts yourself without calling a tool. "
        "For fruit questions, call ask_fruit_specialist. For requests for a "
        "harvest weather pattern, call ask_weather_specialist. If both are "
        "relevant, call both and combine their results. Pass each specialist a "
        "focused sub-question. Never request more than one tool call in a single "
        "response, and never call the same specialist more than once per user "
        "prompt. After any tool calls, give the user the final answer."
    ),
)
