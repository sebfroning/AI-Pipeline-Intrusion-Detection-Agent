import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import MITHRIDAT_MCP_SERVERS, model

MITHRIDAT_SYSTEM_PROMPT = (
    "You are the Mithridatium toolset specialist. You run backdoor and poisoning detection on image-classification models using run_mmbd, run_strip, run_aeva, and run_freeeagle.\n"
    "You receive a focused task from the supervisor with model source, provider, arch, dataset, and which defenses to run. Do not ask the user clarifying questions. If required inputs are missing, return status='needs_input' with missing_fields.\n"
    "Run only the requested defenses. Do not choose a routing strategy unless defenses are unspecified. In that case run mmbd and strip as defaults.\n\n"
    "\nReturn ONLY a JSON object matching this format:\n"
    "specialist, status, model_ref, dataset, findings[], errors[], limitations[]\n"
    "Each finding must include: defense, ok, verdict, metrics, summary.\n"
    "On tool failure (ok=false), record defense, error, and logs in errors[].\n"
    "\nDo not aggregate across defenses, assign overall risk, or write a final report. Do not compare with other specialists. Report raw per-defense results only."
)


async def _build_mithridat_specialist():
    client = MultiServerMCPClient(MITHRIDAT_MCP_SERVERS)
    tools = await client.get_tools()
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=MITHRIDAT_SYSTEM_PROMPT,
    )


def _load_mithridat_specialist():
    return asyncio.run(_build_mithridat_specialist())


mithridat_specialist = _load_mithridat_specialist()
