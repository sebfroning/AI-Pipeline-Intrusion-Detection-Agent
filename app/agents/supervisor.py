from langchain.agents import create_agent
from langchain.tools import tool

from app.agents.detect_specialist import detect_specialist
from app.agents.clean_specialist import clean_specialist
from app.agents.audit_specialist import audit_specialist
from app.config import model


@tool
def invoke_detect_specialist(question: str) -> str:
    """Invoke the detect specialist to use inference-time detection to find backdoors."""
    result = detect_specialist.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return result["messages"][-1].text

@tool
def invoke_clean_specialist(question: str) -> str:
    """Invoke the clean specialist to clean the potentially backdoored model."""
    result = clean_specialist.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return result["messages"][-1].text

@tool
def invoke_audit_specialist(question: str) -> str:
    """Invoke the audit specialist to audit the model's state."""
    result = audit_specialist.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return result["messages"][-1].text

supervisor = create_agent(
    model=model,
    tools=[invoke_detect_specialist, invoke_clean_specialist, invoke_audit_specialist],
    system_prompt=(
        "You are a supervisor with the task of detecting and cleaning backdoors in a pretrained image model."
        "You will be given a prompt and you will need to use the tools available to you to detect and clean the backdoors."
        "The available tools are: invoke_detect_specialist, invoke_clean_specialist, invoke_audit_specialist."
        "You must plan which specialists to invoke and in what order to ensure the model is clean and any backdoors are removed."
        
    ),
)
