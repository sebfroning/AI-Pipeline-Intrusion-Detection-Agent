from langchain.agents import create_agent

from app.config import model
from app.state import OversightState, include_oversight_metadata

CLEAN_SYSTEM_PROMPT = (
    "You are the clean specialist. You clean a potentially backdoored model. "
    "Answer only the focused cleaning task you receive. If required inputs are "
    "missing, say what is missing."
)

clean_specialist = create_agent(
    model=model,
    tools=[],
    system_prompt=CLEAN_SYSTEM_PROMPT,
    middleware=[include_oversight_metadata],
    state_schema=OversightState,
)
