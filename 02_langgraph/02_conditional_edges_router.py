"""
Day 2 · File 02 — Conditional Edges = the Router   (~18 min)

THE BIG IDEA
------------
A normal edge always goes to the same place. A CONDITIONAL edge runs a little
function that LOOKS AT THE STATE and RETURNS THE NAME of the node to go to next.
That function is your router. This is the "router pattern": one place decides,
several specialist nodes handle.

    add_conditional_edges("classify", route_fn, {"math": "math_node", ...})
                            ^source     ^picker   ^map: picker's return -> node

This file routes a user question to the right specialist. We classify with the
LLM (reusing Day 1's structured output), then a pure-Python router sends the
state down one branch. A rule-based variant is shown too, so you can see the
router is just "return a string".

Run:  python 02_langgraph/02_conditional_edges_router.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from providers import get_chat_model


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT — classify, then branch to one of three handlers.
# ─────────────────────────────────────────────────────────────────────────────
class RouterState(TypedDict):
    """Shared state: the question, the chosen route, and the final answer."""
    question: str
    route: str
    answer: str


class Category(BaseModel):
    """The typed classification the router will switch on."""
    kind: Literal["math", "weather", "general"] = Field(
        description="math = arithmetic; weather = asking about weather; general = anything else"
    )


def classify(state: RouterState) -> dict:
    """NODE: ask the model to label the question, store the label in `route`.

    Note we write to state['route'] here; the ROUTER below only reads it. Keep
    'decide the label' (a node) separate from 'pick the edge' (the router fn).
    """
    model = get_chat_model(temperature=0).with_structured_output(Category)
    kind = model.invoke([HumanMessage(state["question"])]).kind
    return {"route": kind}


def route_fn(state: RouterState) -> Literal["math", "weather", "general"]:
    """ROUTER: read the state and RETURN the next node's key. No state change here.

    The return value is matched against the mapping passed to
    add_conditional_edges, which translates it into an actual node name.
    """
    return state["route"]


def math_node(state: RouterState) -> dict:
    """Handler for arithmetic questions."""
    return {"answer": f"[math branch] I'd compute: {state['question']}"}


def weather_node(state: RouterState) -> dict:
    """Handler for weather questions."""
    return {"answer": f"[weather branch] I'd look up weather for: {state['question']}"}


def general_node(state: RouterState) -> dict:
    """Handler for everything else — answer directly with the model."""
    reply = get_chat_model(temperature=0).invoke(
        [HumanMessage(state["question"])]).content
    return {"answer": f"[general branch] {reply}"}


def build_router_graph():
    """START -> classify -> (conditional) -> one handler -> END.

    The conditional edge from 'classify' uses route_fn to choose the branch.
    Each handler then flows straight to END.
    """
    g = StateGraph(RouterState)
    g.add_node("classify", classify)
    g.add_node("math", math_node)
    g.add_node("weather", weather_node)
    g.add_node("general", general_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",           # source node
        route_fn,             # picker: returns "math" | "weather" | "general"
        {"math": "math", "weather": "weather", "general": "general"},  # -> node names
    )
    for handler in ("math", "weather", "general"):
        g.add_edge(handler, END)
    return g.compile()


def demo_concept():
    """Send three different questions through the same graph."""
    app = build_router_graph()
    for q in ["What is 12 * 9?", "Will it rain in Paris today?", "Who wrote Hamlet?"]:
        out = app.invoke({"question": q, "route": "", "answer": ""})
        print(f"Q: {q}\n   route={out['route']!r} -> {out['answer']}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 1 — a RULE-BASED router (no LLM, no mapping dict).
# Take-away: the router is just a function returning a node name. If that name
# IS the node name, you can skip the mapping dict entirely.
# ─────────────────────────────────────────────────────────────────────────────
def demo_rule_based_router():
    """Route by keyword instead of by LLM, returning node names directly."""
    class S(TypedDict):
        question: str
        answer: str

    def picker(s: S) -> str:
        q = s["question"].lower()
        if any(c.isdigit() for c in q):
            return "math"
        if "weather" in q or "rain" in q:
            return "weather"
        return "general"

    g = StateGraph(S)
    g.add_node("math", lambda s: {"answer": "rule->math"})
    g.add_node("weather", lambda s: {"answer": "rule->weather"})
    g.add_node("general", lambda s: {"answer": "rule->general"})
    g.add_edge(START, "math")  # placeholder; replaced below — see note
    # NOTE: a conditional edge must come FROM a node. We add a tiny entry node:
    g = StateGraph(S)
    g.add_node("entry", lambda s: {})            # no-op, just a place to branch from
    g.add_node("math", lambda s: {"answer": "rule->math"})
    g.add_node("weather", lambda s: {"answer": "rule->weather"})
    g.add_node("general", lambda s: {"answer": "rule->general"})
    g.add_edge(START, "entry")
    g.add_conditional_edges("entry", picker)     # no mapping: return == node name
    for h in ("math", "weather", "general"):
        g.add_edge(h, END)

    app = g.compile()
    for q in ["add 2 and 2", "what's the weather?", "tell me a joke"]:
        print(q, "->", app.invoke({"question": q, "answer": ""})["answer"])


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 2 — routing straight to END (early exit).
# Take-away: a branch can return END to stop immediately — handy for "reject",
# "spam", or "nothing to do" paths.
# ─────────────────────────────────────────────────────────────────────────────
def demo_route_to_end():
    """If the message is empty, route directly to END; otherwise handle it."""
    class S(TypedDict):
        question: str
        answer: str

    def picker(s: S):
        return END if not s["question"].strip() else "handle"

    g = StateGraph(S)
    g.add_node("entry", lambda s: {})
    g.add_node("handle", lambda s: {"answer": "handled: " + s["question"]})
    g.add_edge(START, "entry")
    g.add_conditional_edges("entry", picker, {"handle": "handle", END: END})
    g.add_edge("handle", END)

    app = g.compile()
    print("empty   ->", app.invoke({"question": "   ", "answer": "(none)"})["answer"])
    print("nonempty->", app.invoke({"question": "hi", "answer": "(none)"})["answer"])


if __name__ == "__main__":
    print("\n== concept: LLM router ==");           demo_concept()
    print("\n== playing 1: rule-based router ==");   demo_rule_based_router()
    print("\n== playing 2: route to END ==");        demo_route_to_end()
