"""
Day 2 · File 03 — The Agent Loop as a Graph  (THE CENTERPIECE)   (~25 min)

THE BIG IDEA
------------
This is the payoff of the whole week. On Day 1 you used create_agent (the easy
button). Here you BUILD IT BY HAND as a graph with a CYCLE:

        ┌────────────► model ──────────────┐
        │   (calls the LLM with messages)   │
        │                                   ▼
        │                          has tool_calls?  ──no──►  END
        │                                   │yes
        └────────── tools ◄─────────────────┘
            (runs the tools, appends results)

Two new edges make it an agent:
  • a CONDITIONAL edge after `model`: tool_calls? -> "tools", else -> END
  • a normal edge "tools" -> "model" that LOOPS BACK (the cycle)

That cycle — model, tools, model, tools, … until no more tool calls — IS the
ReAct / tool-loop pattern, and IS what create_agent runs for you.

Run:  python 02_langgraph/03_agent_loop_as_graph.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from providers import get_chat_model


# ─────────────────────────────────────────────────────────────────────────────
# Tools the agent may call. The docstring is the model-facing description.
# ─────────────────────────────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    data = {"paris": "Paris: 18°C, light rain",
            "tokyo": "Tokyo: 24°C, clear skies",
            "london": "London: 15°C, foggy"}
    return data.get(city.lower(), f"No weather data for {city}.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '3 * (4 + 5)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) allowed."
    return str(eval(expression, {"__builtins__": {}}, {}))


TOOLS = [get_weather, calculator]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ─────────────────────────────────────────────────────────────────────────────
# State — the conversation. add_messages is the reducer that APPENDS new
# messages instead of overwriting (so the history grows each turn).
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# Bind tools once so the model knows the menu on every call.
_model_with_tools = get_chat_model(temperature=0).bind_tools(TOOLS)


def model_node(state: AgentState) -> dict:
    """NODE: call the LLM with the running message list.

    Returns the model's reply. If the reply contains tool_calls, the router below
    will send us to the tools node; otherwise this reply is the final answer.
    """
    reply = _model_with_tools.invoke(state["messages"])
    return {"messages": [reply]}


def tools_node(state: AgentState) -> dict:
    """NODE: execute every tool the model asked for, append a ToolMessage each.

    This is the part create_agent hides. We read the last AIMessage's tool_calls,
    run each tool ourselves, and feed results back tied to their call id.
    """
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        tool_obj = TOOLS_BY_NAME[call["name"]]
        output = tool_obj.invoke(call["args"])
        results.append(ToolMessage(output, tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: AgentState) -> str:
    """ROUTER: after the model speaks, do we need tools or are we done?

    tool_calls present -> go run them ("tools"); otherwise stop (END).
    """
    return "tools" if state["messages"][-1].tool_calls else END


def build_agent_graph():
    """Assemble the cycle: model -> (tools -> model)* -> END.

    The conditional edge is the decision; the "tools" -> "model" edge is the loop.
    """
    g = StateGraph(AgentState)
    g.add_node("model", model_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "model")
    g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "model")     # <-- the cycle: loop back for another turn
    return g.compile()


def demo_concept():
    """Ask something that needs a tool and watch the loop resolve it."""
    app = build_agent_graph()
    out = app.invoke({"messages": [HumanMessage("What's the weather in Tokyo?")]})
    print("final answer:", out["messages"][-1].content)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 1 — stream the cycle so you SEE model -> tools -> model.
# Take-away: each loop iteration is one model step and (maybe) one tools step.
# ─────────────────────────────────────────────────────────────────────────────
def demo_stream_the_loop():
    """Stream a 2-tool question to watch the cycle turn more than once."""
    app = build_agent_graph()
    q = "What's the weather in London, and what is 15 * 4?"
    for step in app.stream({"messages": [HumanMessage(q)]}, stream_mode="updates"):
        for node, payload in step.items():
            msg = payload["messages"][-1]
            tag = getattr(msg, "tool_calls", None) or msg.content
            print(f"[{node}] -> {tag}")


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 2 — the same thing with PREBUILT helpers.
# Take-away: langgraph ships ToolNode (our tools_node) and tools_condition (our
# should_continue). Swapping them in proves our hand-rolled versions are the
# real thing. This is essentially what create_agent does internally.
# ─────────────────────────────────────────────────────────────────────────────
def demo_with_prebuilt():
    """Rebuild the agent using langgraph.prebuilt instead of hand-rolled nodes."""
    from langgraph.prebuilt import ToolNode, tools_condition

    g = StateGraph(AgentState)
    g.add_node("model", model_node)
    g.add_node("tools", ToolNode(TOOLS))           # <- replaces tools_node
    g.add_edge(START, "model")
    g.add_conditional_edges("model", tools_condition)  # <- replaces should_continue
    g.add_edge("tools", "model")
    app = g.compile()

    out = app.invoke({"messages": [HumanMessage("Weather in Paris?")]})
    print("prebuilt answer:", out["messages"][-1].content)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYING AROUND 3 — print the graph so the shape is undeniable.
# Take-away: you can render any compiled graph as ASCII or Mermaid.
# ─────────────────────────────────────────────────────────────────────────────
def demo_draw_graph():
    """Print the agent graph as ASCII (and the Mermaid source for slides)."""
    app = build_agent_graph()
    try:
        print(app.get_graph().draw_ascii())
    except Exception as e:
        print("(ascii draw needs 'grandalf'; here's Mermaid instead)", e)
    print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    print("\n== concept: the agent loop ==");        demo_concept()
    print("\n== playing 1: stream the cycle ==");     demo_stream_the_loop()
    print("\n== playing 2: prebuilt ToolNode ==");    demo_with_prebuilt()
    print("\n== playing 3: draw the graph ==");       demo_draw_graph()
