"""
capstone/graph.py — wire the nodes into the assistant graph.

This file is deliberately tiny: all the logic lives in nodes.py. Assembly is just
"add nodes, add edges". Read the edges below as a sentence and you have the whole
control flow.

        START
          │
          ▼
       triage ───────────────► route_after_triage (conditional)
                                   │needs_lookup            │else
                                   ▼                        ▼
                              specialist ◄──┐            respond ──► END
                                   │        │               ▲
                          should_continue   │ (loop back)   │
                            │tools     │else└── tools ───────┘
                            ▼          └──────────────► respond
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langgraph.graph import StateGraph, START, END

from state import SupportState
from nodes import (
    triage_node, specialist_node, tools_node, respond_node,
    route_after_triage, should_continue,
)


def build_support_graph(checkpointer=None):
    """Compile and return the support assistant.

    Pass a checkpointer (Day 3) to give it per-conversation memory; leave it None
    for a stateless run. The graph structure is identical either way.
    """
    g = StateGraph(SupportState)

    # The four nodes (see nodes.py for what each does).
    g.add_node("triage", triage_node)
    g.add_node("specialist", specialist_node)
    g.add_node("tools", tools_node)
    g.add_node("respond", respond_node)

    # Sequential entry.
    g.add_edge(START, "triage")

    # ROUTER: after triage, branch on whether we need to look things up.
    g.add_conditional_edges(
        "triage", route_after_triage,
        {"specialist": "specialist", "respond": "respond"},
    )

    # CYCLE: specialist decides to call tools or finish; tools loop back.
    g.add_conditional_edges(
        "specialist", should_continue,
        {"tools": "tools", "respond": "respond"},
    )
    g.add_edge("tools", "specialist")     # loop back for another model step

    # Sequential exit.
    g.add_edge("respond", END)

    return g.compile(checkpointer=checkpointer)
