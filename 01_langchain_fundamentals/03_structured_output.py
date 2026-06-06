"""
3. Structured output  (~10 min)

with_structured_output(Schema) forces the model to return a VALIDATED Pydantic
object instead of free text. It turns "text you'd have to parse and pray" into
"an object your code can branch on".

This is the bridge from "the model talks" to "the model drives control flow":
  - tomorrow, a LangGraph router routes on a typed classification like this
  - the course project's verdict node emits exactly a structured object like this

Run:  python 01_langchain_fundamentals/03_structured_output.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from providers import get_chat_model


# A generic, relatable schema: triage an incoming support message.
# (Note how this foreshadows Day 2 routing -- category is a routing key.)
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
