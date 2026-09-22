from langgraph.graph import StateGraph, START, END
from app.state import AgentState
from app.nodes import supervisor, detector, auditor, cleaner, reporter

g = StateGraph(AgentState)
g.add_node("supervisor", supervisor)
g.add_node("detector", detector)
g.add_node("auditor", auditor)
g.add_node("cleaner", cleaner)
g.add_node("reporter", reporter)

g.add_edge(START, "supervisor")
g.add_edge("detector", "supervisor")
g.add_edge("auditor", "supervisor")
g.add_edge("cleaner", "supervisor")

g.add_edge("reporter", END)

graph = g.compile()