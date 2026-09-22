import random

from langchain.agents import create_agent
from langchain.tools import tool

from app.config import model


@tool
def harvest_weather() -> str:
    """Choose a weather pattern that may affect a fruit harvest."""
    return random.choice(["sunny", "rainy", "dry"])


weather_specialist = create_agent(
    model=model,
    tools=[harvest_weather],
    system_prompt=(
        "You are a harvest-weather specialist. Reply with exactly one of these "
        "labels and nothing else: sunny, rainy, or dry. You may use the "
        "harvest_weather tool to choose the label."
    ),
)
