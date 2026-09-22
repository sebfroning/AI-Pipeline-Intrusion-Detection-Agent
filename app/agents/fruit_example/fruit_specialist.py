from langchain.agents import create_agent
from langchain.tools import tool

from app.config import model


@tool
def fruit_info(fruit_name: str) -> str:
    """Look up basic information about a fruit."""
    return f"Info about {fruit_name}"


fruit_specialist = create_agent(
    model=model,
    tools=[fruit_info],
    system_prompt=(
        "You are a fruit specialist. Answer only the focused fruit question you "
        "receive. You may use fruit_info when it is helpful. Keep your answer concise."
    ),
)
