"""
Day 3 · File 04 — Short-term memory: checkpointer + thread_id   (~15 min)

THE BIG IDEA
------------
By default a graph forgets everything between .invoke() calls. A CHECKPOINTER
saves the State after every step, keyed by a THREAD_ID you pass in the config.
Re-invoke with the same thread_id and the graph resumes with its memory intact —
that's a conversation. Use a different thread_id and you get a clean slate.

    app = graph.compile(checkpointer=InMemorySaver())
    app.invoke(input, {"configurable": {"thread_id": "alice"}})   # remembers "alice"

This is "short-term" memory: per-conversation, lives as long as the checkpointer
does. (Cross-conversation memory is File 05's job.)

Run:  python 03_memory/04_short_term_memory.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from providers import get_chat_model


class ChatState(TypedDict):
    """Just the running conversation; add_messages appends each turn."""
    messages: Annotated[list, add_messages]


_model = get_chat_model(temperature=0)


def chat_node(state: ChatState) -> dict:
    """NODE: reply using the FULL message history in state.

    The node itself is stateless — the memory comes entirely from the checkpointer
    re-loading prior messages into `state` before this runs.
    """
    return {"messages": [_model.invoke(state["messages"])]}


def build_chatbot():
    """A one-node chat graph, compiled WITH a checkpointer so it can remember."""
    g = StateGraph(ChatState)
    g.add_node("chat", chat_node)
    g.add_edge(START, "chat")
    g.add_edge("chat", END)
    return g.compile(checkpointer=InMemorySaver())


def say(app, thread_id: str, text: str) -> str:
    """Helper: send one message on a given thread and return the reply text."""
    cfg = {"configurable": {"thread_id": thread_id}}
    out = app.invoke({"messages": [HumanMessage(text)]}, cfg)
    return out["messages"][-1].content


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT — same thread_id remembers across calls.
# ─────────────────────────────────────────────────────────────────────────────
def demo_concept():
    """Tell it your name, then ask for it back — on the same thread."""
    app = build_chatbot()
    print("turn 1:", say(app, "alice", "Hi! My name is Ada and I love Python."))
    print("turn 2:", say(app, "alice", "What's my name and what do I love?"))
    # It answers correctly because thread 'alice' still holds turn 1.


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 1 — thread isolation. A different thread_id can't see it.
# Take-away: thread_id IS the conversation boundary.
# ─────────────────────────────────────────────────────────────────────────────
def demo_thread_isolation():
    """Ask 'bob' (a fresh thread) what 'alice' said — it has no idea."""
    app = build_chatbot()
    say(app, "alice", "Remember the secret word is 'banana'.")
    print("alice:", say(app, "alice", "What's the secret word?"))
    print("bob:  ", say(app, "bob", "What's the secret word?"))   # no memory of it


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 2 — inspect the saved state.
# Take-away: get_state() lets you read what the checkpointer is holding.
# ─────────────────────────────────────────────────────────────────────────────
def demo_inspect_state():
    """Peek at how many messages the checkpointer has stored for a thread."""
    app = build_chatbot()
    say(app, "carol", "Message one.")
    say(app, "carol", "Message two.")
    snap = app.get_state({"configurable": {"thread_id": "carol"}})
    print("stored messages on 'carol':", len(snap.values["messages"]))
    for m in snap.values["messages"]:
        print("  -", type(m).__name__, ":", m.content[:50])


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 3 — time travel: list checkpoints (history).
# Take-away: every step is checkpointed, so you can see (and resume from) the past.
# ─────────────────────────────────────────────────────────────────────────────
def demo_history():
    """List the checkpoints recorded for a thread, newest first."""
    app = build_chatbot()
    say(app, "dave", "First.")
    say(app, "dave", "Second.")
    cfg = {"configurable": {"thread_id": "dave"}}
    states = list(app.get_state_history(cfg))
    print(f"'dave' has {len(states)} checkpoints (each step is one).")


if __name__ == "__main__":
    print("\n== concept: same-thread memory ==");   demo_concept()
    print("\n== playing 1: thread isolation ==");    demo_thread_isolation()
    print("\n== playing 2: inspect state ==");       demo_inspect_state()
    print("\n== playing 3: checkpoint history ==");  demo_history()
