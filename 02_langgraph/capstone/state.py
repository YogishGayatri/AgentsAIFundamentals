"""
capstone/state.py — the shared State for the whole assistant.

This single dict is the assistant's working memory as it flows through the graph.
`messages` uses the add_messages reducer (append, don't overwrite) so the
conversation + any tool results accumulate during the specialist cycle. The
other fields are plain (overwrite) values that each node fills in along the way.
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class SupportState(TypedDict):
    """End-to-end state for one support request.

    Filled in roughly in this order as the graph runs:
        triage     -> category, urgency, summary, needs_lookup, escalated
        specialist -> appends AIMessage(s) and ToolMessage(s) to `messages`
        respond    -> resolution (the final customer-facing reply)
    """
    messages: Annotated[list, add_messages]   # conversation, grows via the cycle
    category: str                             # set by triage
    urgency: int                              # set by triage
    summary: str                              # set by triage
    needs_lookup: bool                        # set by triage -> drives the router
    escalated: bool                           # set by triage (urgency >= 4)
    resolution: str                           # set by respond (final answer)
