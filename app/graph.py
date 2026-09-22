from langgraph.graph import StateGraph, START, END
from app.state import AgentState, supervisor
from app.agents.detect_specialist import detect_specialist
from app.agents.audit_specialist import audit_specialist
from app.agents.clean_specialist import clean_specialist
from app.agents.reporter import reporter

g = StateGraph(AgentState)
g.add_node("supervisor", supervisor)
g.add_node("detector", detect_specialist)
g.add_node("auditor", audit_specialist)
g.add_node("cleaner", clean_specialist)
g.add_node("reporter", reporter)

g.add_edge(START, "supervisor")
g.add_edge("detector", "supervisor")
g.add_edge("auditor", "supervisor")
g.add_edge("cleaner", "supervisor")

g.add_edge("reporter", END)

graph = g.compile()