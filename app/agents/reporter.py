import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import MITHRIDAT_MCP_SERVERS, model

REPORTER_SYSTEM_PROMPT = (
    "You are the reporter. You are responsible for reporting and explaining the results of the ML detection and cleaning agent pipeline."
    "You read the shared memory of the specialist agents and report the results to the user."
)


async def _build_reporter():
    return create_agent(
        model=model
    )


def _load_reporter():
    return asyncio.run(_build_reporter())


reporter = _load_reporter()
