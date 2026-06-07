"""
Day 3 · File 05 — Long-term memory: the Store (cross-thread)   (~12 min)

THE BIG IDEA
------------
A checkpointer remembers ONE conversation (one thread_id). A STORE remembers
facts ACROSS conversations — keyed by something durable like a user_id. That's
how an assistant recalls "you prefer Python" in a brand-new chat next week.

    store = InMemoryStore()
    store.put(("memories", user_id), key, {"text": "..."})   # write a fact
    store.search(("memories", user_id))                      # read them back

A node gets the store injected just by declaring a `store` parameter. The
namespace tuple (e.g. ("memories", user_id)) keeps each user's facts separate.

Run:  python 03_memory/05_long_term_memory.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.messages import HumanMessage, SystemMessage
from providers import get_chat_model


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


_model = get_chat_model(temperature=0)


def agent_node(state: ChatState, *, store: BaseStore, config) -> dict:
    """NODE: recall this user's saved facts from the store, then reply.

    `store` and `config` are injected by LangGraph (because we named the params).
    We look up the user's namespace, fold any remembered facts into the system
    prompt, and — as a tiny demo — also SAVE the latest user message as a memory.
    """
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    # READ: pull everything we remember about this user (across all threads).
    remembered = [item.value["text"] for item in store.search(namespace)]
    memory_note = ("Known facts about the user: " + "; ".join(remembered)) if remembered \
        else "No saved facts about the user yet."

    reply = _model.invoke(
        [SystemMessage("You are a helpful assistant. " + memory_note)] + state["messages"]
    )

    # WRITE: remember the latest thing the user said (naive but illustrative).
    store.put(namespace, str(uuid.uuid4()), {"text": state["messages"][-1].content})

    return {"messages": [reply]}


def build_agent():
    """Compile WITH both a store (cross-thread) and a checkpointer (per-thread)."""
    g = StateGraph(ChatState)
    g.add_node("agent", agent_node)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile(checkpointer=InMemorySaver(), store=InMemoryStore())


def ask(app, user_id: str, thread_id: str, text: str) -> str:
    """Helper: one turn, scoped to a user_id (store) AND a thread_id (checkpointer)."""
    cfg = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
    return app.invoke({"messages": [HumanMessage(text)]}, cfg)["messages"][-1].content


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT — a fact learned in one thread is recalled in a DIFFERENT thread.
# Take-away: the store crosses conversations; the checkpointer does not.
# ─────────────────────────────────────────────────────────────────────────────
def demo_concept():
    """User 'u1' shares a fact in thread A, then we recall it in thread B."""
    app = build_agent()
    print("thread A:", ask(app, "u1", "threadA", "I'm a data scientist who loves Python."))
    # New thread, SAME user -> the store still has the fact from thread A.
    print("thread B:", ask(app, "u1", "threadB", "Based on what you know, suggest a hobby for me."))


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 1 — per-user isolation.
# Take-away: a different user_id is a different namespace = different memories.
# ─────────────────────────────────────────────────────────────────────────────
def demo_user_isolation():
    """u1's facts are invisible to u2."""
    app = build_agent()
    ask(app, "u1", "t1", "My favorite color is teal.")
    print("u2 (different user):", ask(app, "u2", "t2", "What's my favorite color?"))


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 2 — the raw store API, no graph involved.
# Take-away: put / get / search are just a namespaced key-value store.
# ─────────────────────────────────────────────────────────────────────────────
def demo_raw_store_api():
    """Use InMemoryStore directly to see exactly what the node is doing."""
    store = InMemoryStore()
    ns = ("memories", "u1")
    store.put(ns, "fact1", {"text": "likes hiking"})
    store.put(ns, "fact2", {"text": "lives in Berlin"})
    print("get fact1:", store.get(ns, "fact1").value)
    print("search all:", [i.value["text"] for i in store.search(ns)])


if __name__ == "__main__":
    print("\n== concept: cross-thread recall ==");   demo_concept()
    print("\n== playing 1: per-user isolation ==");   demo_user_isolation()
    print("\n== playing 2: raw store API ==");        demo_raw_store_api()
