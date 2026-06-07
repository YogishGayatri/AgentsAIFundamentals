"""
Day 2 · File 01 — State, Nodes, Edges, Reducers (sequential graph)   (~25 min)

THE BIG IDEA
------------
LangGraph turns your agent into an explicit graph:
  • STATE   a shared dict that every step reads from and writes to.
  • NODE    a plain function: state in -> a PARTIAL state update out.
  • EDGE    a wire saying "after node A, go to node B".
  • REDUCER the rule for HOW an update merges into the state. Default rule is
            "overwrite". A reducer like `operator.add` says "append instead".

This file builds the simplest graph there is — a straight line (sequential
edges) — so you can see those four pieces in isolation. Tomorrow's router and
the agent loop are just this with smarter edges.

Run:  python 02_langgraph/01_state_nodes_reducers.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT — a sequential pipeline of three nodes sharing one State.
# ─────────────────────────────────────────────────────────────────────────────
class PipelineState(TypedDict):
    """The shared memory of the graph.

    `text` uses the DEFAULT reducer (overwrite): each node that returns `text`
    replaces it. `log` uses `operator.add` so every node APPENDS one line and
    nothing is lost — that's a reducer doing its job.
    """
    text: str
    log: Annotated[list[str], operator.add]


def clean(state: PipelineState) -> dict:
    """NODE: strip whitespace and lowercase the text.

    A node receives the whole state and returns ONLY the keys it wants to change.
    """
    return {"text": state["text"].strip().lower(), "log": ["cleaned"]}


def exclaim(state: PipelineState) -> dict:
    """NODE: add excitement. Overwrites `text`, appends to `log`."""
    return {"text": state["text"] + "!!!", "log": ["exclaimed"]}


def shout(state: PipelineState) -> dict:
    """NODE: uppercase the whole thing."""
    return {"text": state["text"].upper(), "log": ["shouted"]}


def build_sequential_graph():
    """Wire three nodes in a straight line: START -> clean -> exclaim -> shout -> END.

    add_edge(A, B) is a SEQUENTIAL edge. START and END are the built-in entry and
    exit. .compile() freezes the graph into a runnable you can .invoke()/.stream().
    """
    g = StateGraph(PipelineState)
    g.add_node("clean", clean)
    g.add_node("exclaim", exclaim)
    g.add_node("shout", shout)

    g.add_edge(START, "clean")
    g.add_edge("clean", "exclaim")
    g.add_edge("exclaim", "shout")
    g.add_edge("shout", END)
    return g.compile()


def demo_concept():
    """Run the pipeline once and watch the state evolve."""
    app = build_sequential_graph()
    result = app.invoke({"text": "  Hello World  ", "log": []})
    print("final text:", result["text"])
    print("log (accumulated by the reducer):", result["log"])


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 1 — overwrite vs. reducer, shown side by side.
# Take-away: WITHOUT a reducer, parallel/repeated writes clobber each other;
# WITH one, they combine.
# ─────────────────────────────────────────────────────────────────────────────
def demo_overwrite_vs_reducer():
    """Compare a plain field to a reduced field across two nodes."""
    class S(TypedDict):
        last: str                                # overwrite (default)
        all: Annotated[list[str], operator.add]  # append (reducer)

    g = StateGraph(S)
    g.add_node("n1", lambda s: {"last": "a", "all": ["a"]})
    g.add_node("n2", lambda s: {"last": "b", "all": ["b"]})
    g.add_edge(START, "n1"); g.add_edge("n1", "n2"); g.add_edge("n2", END)

    out = g.compile().invoke({"last": "", "all": []})
    print("last (overwritten -> only n2 survives):", out["last"])
    print("all  (reduced     -> both kept):       ", out["all"])


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 2 — stream the graph to watch it node-by-node.
# Take-away: stream_mode='updates' yields one entry per node as it fires.
# ─────────────────────────────────────────────────────────────────────────────
def demo_streaming():
    """Stream the sequential graph so you SEE each node fire in order."""
    app = build_sequential_graph()
    for step in app.stream({"text": "  Stream Me  ", "log": []}, stream_mode="updates"):
        print("step ->", step)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 3 — a custom reducer (your own merge rule).
# Take-away: a reducer is just a function (old_value, new_value) -> merged_value.
# Here we keep a running maximum instead of appending.
# ─────────────────────────────────────────────────────────────────────────────
def demo_custom_reducer():
    """Use a custom reducer that keeps the largest number seen."""
    def keep_max(old: int, new: int) -> int:
        return max(old, new)

    class S(TypedDict):
        high_score: Annotated[int, keep_max]

    g = StateGraph(S)
    g.add_node("p1", lambda s: {"high_score": 30})
    g.add_node("p2", lambda s: {"high_score": 10})   # smaller -> ignored by reducer
    g.add_node("p3", lambda s: {"high_score": 50})   # bigger  -> wins
    g.add_edge(START, "p1"); g.add_edge("p1", "p2")
    g.add_edge("p2", "p3"); g.add_edge("p3", END)

    print("high_score (custom max reducer):", g.compile().invoke({"high_score": 0}))


if __name__ == "__main__":
    print("\n== concept: sequential pipeline ==");      demo_concept()
    print("\n== playing 1: overwrite vs reducer ==");    demo_overwrite_vs_reducer()
    print("\n== playing 2: streaming the nodes ==");     demo_streaming()
    print("\n== playing 3: a custom reducer ==");        demo_custom_reducer()
