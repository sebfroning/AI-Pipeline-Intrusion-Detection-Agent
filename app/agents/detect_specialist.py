from langchain.agents import create_agent

from app.config import model
from app.state import OversightState, include_oversight_metadata

DETECT_SYSTEM_PROMPT = (
    "You are the detect specialist. You use inference-time detection to find "
    "backdoors in a pretrained image model. Answer only the focused detection "
    "task you receive. If required inputs are missing, say what is missing."
)

detect_specialist = create_agent(
    model=model,
    tools=[],
    system_prompt=DETECT_SYSTEM_PROMPT,
    middleware=[include_oversight_metadata],
    state_schema=OversightState,
)
