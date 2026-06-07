"""
Day 3 · File 06 — Human-in-the-loop: pause / approve / resume   (~8 min, optional)

THE BIG IDEA
------------
Some actions shouldn't happen without a human OK (refunds, deletes, sending
email). LangGraph lets a node PAUSE the graph mid-run, hand control back to you,
and RESUME exactly where it left off once you decide.

    decision = interrupt({"question": "Approve refund of $20?"})   # graph pauses here
    ...later...
    app.invoke(Command(resume="approve"), config)                 # graph continues

Requirements: the graph must be compiled with a checkpointer (so it can save the
paused state) and you must call it with a thread_id.

Run:  python 03_memory/06_human_in_the_loop.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class ActionState(TypedDict):
    """A proposed action and the outcome after a human decides."""
    action: str
    outcome: str


def propose(state: ActionState) -> dict:
    """NODE: (pretend the agent decided to do something risky.) Just pass it along."""
    return {}


def approval_gate(state: ActionState) -> dict:
    """NODE: PAUSE and ask a human to approve the action.

    interrupt(payload) stops the graph and surfaces `payload` to the caller. The
    value you later pass as Command(resume=...) becomes interrupt()'s return value.
    """
    decision = interrupt({"question": f"Approve action: '{state['action']}'?",
                          "options": ["approve", "reject"]})
    if decision == "approve":
        return {"outcome": f"✅ Executed: {state['action']}"}
    return {"outcome": f"🚫 Cancelled: {state['action']}"}


def build_graph():
    """propose -> approval_gate -> END, compiled with a checkpointer (required)."""
    g = StateGraph(ActionState)
    g.add_node("propose", propose)
    g.add_node("approval_gate", approval_gate)
    g.add_edge(START, "propose")
    g.add_edge("propose", "approval_gate")
    g.add_edge("approval_gate", END)
    return g.compile(checkpointer=InMemorySaver())


def run_until_pause(app, thread_id: str, action: str):
    """Start the graph; it will pause at the approval gate. Return the question."""
    cfg = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"action": action, "outcome": ""}, cfg)
    # When a graph interrupts, the pending question(s) show up under __interrupt__.
    return result["__interrupt__"][0].value, cfg


def resume(app, cfg, decision: str) -> str:
    """Resume the paused graph with a human decision; return the final outcome."""
    return app.invoke(Command(resume=decision), cfg)["outcome"]


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT — pause, then approve.
# ─────────────────────────────────────────────────────────────────────────────
def demo_approve():
    """Run -> pause -> approve -> action executes."""
    app = build_graph()
    question, cfg = run_until_pause(app, "t-approve", "Refund $20 to ACC-9")
    print("PAUSED, asking:", question)
    print("RESULT:", resume(app, cfg, "approve"))


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 1 — the reject path.
# Take-away: the SAME pause, a different resume value, the opposite outcome.
# ─────────────────────────────────────────────────────────────────────────────
def demo_reject():
    """Run -> pause -> reject -> action is cancelled."""
    app = build_graph()
    question, cfg = run_until_pause(app, "t-reject", "Delete account ACC-9")
    print("PAUSED, asking:", question)
    print("RESULT:", resume(app, cfg, "reject"))


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 2 — interactive approval at the terminal.
# Take-away: the "human" can be anything — a person, an API, another system.
# ─────────────────────────────────────────────────────────────────────────────
def demo_interactive():
    """Ask YOU to approve at the prompt."""
    app = build_graph()
    question, cfg = run_until_pause(app, "t-interactive", "Send password reset email")
    print("PAUSED, asking:", question)
    decision = input("type approve/reject> ").strip() or "reject"
    print("RESULT:", resume(app, cfg, decision))


if __name__ == "__main__":
    print("\n== concept: approve ==");        demo_approve()
    print("\n== playing 1: reject ==");        demo_reject()
    print("\n== playing 2: interactive ==");   demo_interactive()
