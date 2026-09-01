import os
import sys
from pathlib import Path

from langchain_ollama import ChatOllama


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

model = ChatOllama(model=OLLAMA_MODEL)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MITHRIDAT_MCP_ROOT = Path(
    os.getenv("MITHRIDAT_MCP_ROOT", str(_REPO_ROOT.parent / "MithridatMCP"))
)

MITHRIDAT_MCP_COMMAND = os.getenv("MITHRIDAT_MCP_COMMAND", sys.executable)
MITHRIDAT_MCP_ARGS = os.getenv(
    "MITHRIDAT_MCP_ARGS",
    "-m,mithridatmcp.mcp_server",
).split(",")


def _mithridat_mcp_env() -> dict[str, str]:
    env = dict(os.environ)
    if _MITHRIDAT_MCP_ROOT.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{_MITHRIDAT_MCP_ROOT}{os.pathsep}{existing}"
            if existing
            else str(_MITHRIDAT_MCP_ROOT)
        )
    return env


MITHRIDAT_MCP_SERVERS = {
    "mithridat": {
        "transport": "stdio",
        "command": MITHRIDAT_MCP_COMMAND,
        "args": MITHRIDAT_MCP_ARGS,
        "env": _mithridat_mcp_env(),
    }
}
