"""
capstone/main.py — run the Support Triage & Resolution Assistant.

Three ways to use it:
    python 02_langgraph/capstone/main.py            # run the sample tickets
    python 02_langgraph/capstone/main.py --chat     # type your own tickets
    python 02_langgraph/capstone/main.py --draw     # print the graph diagram

Watch the trace: triage -> (router) -> specialist <-> tools (the cycle) -> respond.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_core.messages import HumanMessage
from graph import build_support_graph


# A spread of tickets that exercises every path through the graph.
SAMPLE_TICKETS = [
    "I was charged twice for ACC-9 this month — I need the duplicate refunded ASAP!",
    "Where is my order A1234? It was supposed to arrive yesterday.",
    "How do I reset my password?",
    "Just wanted to say your product is great, thanks!",   # general, no lookup
]


def run_ticket(app, text: str):
    """Stream one ticket through the graph and print the live trace + result."""
    print("\n" + "=" * 70)
    print("TICKET:", text)
    print("-" * 70)

    final_state = None
    for step in app.stream({"messages": [HumanMessage(text)]}, stream_mode="updates"):
        for node, payload in step.items():
            if node == "triage":
                print(f"  [triage]     category={payload['category']} "
                      f"urgency={payload['urgency']} needs_lookup={payload['needs_lookup']}")
            elif node == "specialist":
                calls = payload["messages"][-1].tool_calls
                print(f"  [specialist] {'requested ' + str([c['name'] for c in calls]) if calls else 'no tools, finishing'}")
            elif node == "tools":
                for m in payload["messages"]:
                    print(f"  [tools]      -> {m.content}")
            elif node == "respond":
                final_state = payload

    print("-" * 70)
    print("RESOLUTION:\n" + (final_state["resolution"] if final_state else "(none)"))


def main():
    args = sys.argv[1:]

    if "--draw" in args:
        app = build_support_graph()
        print(app.get_graph().draw_mermaid())
        return

    app = build_support_graph()

    if "--chat" in args:
        print("Support assistant. Type a ticket (or 'exit').")
        while True:
            try:
                text = input("\nticket> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"exit", "quit"} or not text:
                break
            run_ticket(app, text)
        return

    for ticket in SAMPLE_TICKETS:
        run_ticket(app, ticket)


if __name__ == "__main__":
    main()
