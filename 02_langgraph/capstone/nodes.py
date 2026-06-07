"""
capstone/nodes.py — every node and router function for the assistant.

Read this top-to-bottom and you've read the whole brain of the project. Each
node is "state in -> partial state out"; each router is "state in -> next-node
name out". The graph.py file only wires these together.

The three Day-2 mechanics all live here:
  • sequential : triage -> ... -> respond
  • router     : route_after_triage  (conditional edge)
  • cycle      : specialist <-> tools (conditional edge + loop-back)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import END
from providers import get_chat_model

from schemas import Triage
from state import SupportState
from tools import SUPPORT_TOOLS, TOOLS_BY_NAME


# Bind the tools once; the specialist reuses this on every loop iteration.
_specialist_model = get_chat_model(temperature=0).bind_tools(SUPPORT_TOOLS)


def _last_user_text(state: SupportState) -> str:
    """Helper: pull the most recent human message text out of the state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return state["messages"][-1].content


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — triage. Classify the request into the typed Triage schema and copy
# its fields onto the state. This is the "understand before you act" step.
# ─────────────────────────────────────────────────────────────────────────────
def triage_node(state: SupportState) -> dict:
    """Classify the incoming message; set category/urgency/summary/needs_lookup.

    Also raises the `escalated` flag for high-urgency tickets (>= 4) so the final
    response can lead with an escalation banner.
    """
    model = get_chat_model(temperature=0).with_structured_output(Triage)
    t: Triage = model.invoke([
        SystemMessage("You triage customer support messages. Be decisive."),
        HumanMessage(_last_user_text(state)),
    ])
    return {
        "category": t.category,
        "urgency": t.urgency,
        "summary": t.summary,
        "needs_lookup": t.needs_lookup,
        "escalated": t.urgency >= 4,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER 1 — after triage, decide: do we need to look things up (enter the
# tool cycle) or can we answer directly?
# ─────────────────────────────────────────────────────────────────────────────
def route_after_triage(state: SupportState) -> str:
    """Return 'specialist' if the request needs data lookups, else 'respond'."""
    return "specialist" if state["needs_lookup"] else "respond"


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — specialist (the model half of the cycle). Calls the LLM with tools
# bound, guided by a category-aware system prompt.
# ─────────────────────────────────────────────────────────────────────────────
def specialist_node(state: SupportState) -> dict:
    """Run one model step that may request tools to resolve the ticket."""
    system = SystemMessage(
        f"You are a support specialist handling a '{state['category']}' issue. "
        f"Use the available tools to fetch real data before answering. "
        f"Once you have what you need, stop calling tools."
    )
    reply = _specialist_model.invoke([system] + state["messages"])
    return {"messages": [reply]}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER 2 — the cycle's decision: did the specialist ask for tools?
# ─────────────────────────────────────────────────────────────────────────────
def should_continue(state: SupportState) -> str:
    """tool_calls present -> 'tools' (run them); otherwise -> 'respond' (we're done)."""
    return "tools" if state["messages"][-1].tool_calls else "respond"


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — tools (the other half of the cycle). Execute every requested tool.
# ─────────────────────────────────────────────────────────────────────────────
def tools_node(state: SupportState) -> dict:
    """Execute the specialist's tool calls and append a ToolMessage for each."""
    results = []
    for call in state["messages"][-1].tool_calls:
        output = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        results.append(ToolMessage(output, tool_call_id=call["id"]))
    return {"messages": results}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — respond. Compose the final customer-facing reply from everything we
# learned, prepending an escalation banner for urgent tickets.
# ─────────────────────────────────────────────────────────────────────────────
def respond_node(state: SupportState) -> dict:
    """Write the final, polished resolution and store it in `resolution`."""
    model = get_chat_model(temperature=0)
    answer = model.invoke(
        [SystemMessage(
            "You are a friendly support agent. Using the conversation and any tool "
            "results above, write a short, clear reply to the customer. "
            f"Issue summary: {state['summary']}.")
        ] + state["messages"]
    ).content

    if state.get("escalated"):
        answer = ("⚠️ [URGENT — flagged for a human agent]\n" + answer)

    return {"resolution": answer, "messages": [AIMessage(answer)]}
