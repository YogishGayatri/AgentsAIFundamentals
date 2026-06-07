"""
capstone/schemas.py — the typed contracts the graph relies on.

We keep Pydantic schemas in one place so every node agrees on the same shape.
`Triage` is what the triage node produces via structured output; the router then
switches on its fields. This is the Day-1 "structured output drives control flow"
idea, now powering a real graph.
"""

from typing import Literal
from pydantic import BaseModel, Field


class Triage(BaseModel):
    """The classification of an incoming support message.

    Every field here is a decision the graph will act on:
      • category    -> which specialist / knowledge a branch should use
      • urgency     -> whether we raise an escalation banner
      • needs_lookup-> whether we enter the tool-using cycle or answer directly
      • summary     -> a one-liner the final response is built from
    """
    category: Literal["billing", "technical", "account", "general"] = Field(
        description="billing=charges/refunds; technical=bugs/errors; "
                    "account=login/profile/orders; general=anything else"
    )
    urgency: int = Field(description="1 (calm) to 5 (critical / customer is blocked)")
    needs_lookup: bool = Field(
        description="True if answering requires looking up order/billing/account "
                    "data or the help center; False if it can be answered directly."
    )
    summary: str = Field(description="One concise sentence describing the request.")
