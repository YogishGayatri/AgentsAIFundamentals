"""
3. Structured output 
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from providers import get_chat_model


# A generic, relatable schema: triage an incoming support message.
class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "general"] = Field(
        description="Which queue this message belongs in"
    )
    urgency: int = Field(description="How urgent, from 1 (low) to 5 (critical)")
    summary: str = Field(description="One-line summary of what the user wants")


model = get_chat_model(temperature=0)
structured_model = model.with_structured_output(TicketClassification)

message = (
    "Hi, I was charged twice for my subscription this month and I need the "
    "duplicate refunded before my card statement closes tomorrow!"
)

result = structured_model.invoke([HumanMessage(message)])

# `result` is a real, validated Python object -- not a string you have to parse.
print("type:    ", type(result).__name__)
print("category:", result.category)
print("urgency: ", result.urgency)
print("summary: ", result.summary)

# Because it's typed, your code can branch on it directly:
if result.category == "billing" and result.urgency >= 4:
    print("\n-> Route: escalate to the billing team immediately.")
