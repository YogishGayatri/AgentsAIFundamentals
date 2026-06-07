"""
03 - Structured output (LangChain)

Big idea: `with_structured_output()` forces the model to return a *validated
Pydantic object* instead of free text. That turns "the model talks" into
"the model drives control flow" -- the typed field becomes a routing decision,
which is exactly what a LangGraph conditional edge will branch on later.

Each demo is its own function. Edit main() at the bottom to show only what you
want. Requires a `providers.py` one level up that exposes get_chat_model(), and
a GROQ_API_KEY in your .env.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_core.messages import HumanMessage
from providers import get_chat_model

model = get_chat_model(temperature=0)

SAMPLE = (
    "Hi, I was charged twice for my subscription this month and I need the "
    "duplicate refunded before my card statement closes tomorrow!"
)
VAGUE = "hey can you help me out?"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TicketVague(BaseModel):
    # No field descriptions on purpose -- compare against TicketClassification.
    category: Literal["billing", "technical", "general"]
    urgency: int
    summary: str


class TicketClassification(BaseModel):
    """Triage an incoming customer support message into a queue."""
    category: Literal["billing", "technical", "general"] = Field(
        description="billing = payments/refunds; technical = bugs/errors; general = everything else"
    )
    urgency: int = Field(ge=1, le=5, description="1 (can wait) to 5 (revenue/security impact, act now)")
    summary: str = Field(description="One line, under 12 words, in the user's own framing")

    @field_validator("urgency")
    @classmethod
    def _in_range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("urgency must be between 1 and 5")
        return v


class TicketTriage(BaseModel):
    """Triage a message; use 'unknown' and null when there isn't enough info."""
    category: Literal["billing", "technical", "general", "unknown"]
    urgency: int = Field(ge=1, le=5)
    summary: str
    account_id: Optional[str] = Field(default=None, description="Account/order ID if mentioned, else null")


class Entity(BaseModel):
    text: str
    kind: Literal["product", "amount", "date"]


class TicketAnalysis(BaseModel):
    """Classify the ticket and extract key entities in a single call."""
    category: Literal["billing", "technical", "general"]
    urgency: int = Field(ge=1, le=5)
    entities: list[Entity] = Field(description="Key entities mentioned in the message")


# ---------------------------------------------------------------------------
# 1. Basic: a validated object, not a string you have to parse
# ---------------------------------------------------------------------------
def demo_basic():
    structured = model.with_structured_output(TicketClassification)
    r = structured.invoke([HumanMessage(SAMPLE)])
    print("type:    ", type(r).__name__)
    print("category:", r.category)
    print("urgency: ", r.urgency)
    print("summary: ", r.summary)


# ---------------------------------------------------------------------------
# 2. Field descriptions steer the model (same input, two schemas)
# ---------------------------------------------------------------------------
def demo_descriptions():
    vague = model.with_structured_output(TicketVague).invoke([HumanMessage(SAMPLE)])
    rich = model.with_structured_output(TicketClassification).invoke([HumanMessage(SAMPLE)])
    print("vague schema ->", vague.model_dump())
    print("rich schema  ->", rich.model_dump())
    print("(the gap shows most on ambiguous messages -- descriptions are instructions)")


# ---------------------------------------------------------------------------
# 3. The typed field IS a router (the Day-2 bridge)
# ---------------------------------------------------------------------------
def demo_router():
    t = model.with_structured_output(TicketClassification).invoke([HumanMessage(SAMPLE)])
    handlers = {
        "billing":   lambda x: f"-> Billing queue  (P{x.urgency}): {x.summary}",
        "technical": lambda x: f"-> Eng queue      (P{x.urgency}): {x.summary}",
        "general":   lambda x: f"-> General queue  (P{x.urgency}): {x.summary}",
    }
    print(handlers[t.category](t))
    print("(a typed field -> a dispatch key; in LangGraph this becomes a conditional edge)")


# ---------------------------------------------------------------------------
# 4. include_raw=True: structured output is really a *forced tool call*
# ---------------------------------------------------------------------------
def demo_include_raw():
    structured = model.with_structured_output(TicketClassification, include_raw=True)
    out = structured.invoke([HumanMessage(SAMPLE)])
    print("raw tool call:", out["raw"].tool_calls)   # the schema was called as a tool
    print("parsed:       ", out["parsed"])
    print("parse error:  ", out["parsing_error"])


# ---------------------------------------------------------------------------
# 5. Constraints validate the result -- and how to handle a bad parse safely
# ---------------------------------------------------------------------------
def demo_validation():
    # Prove what *would* happen if the model returned an out-of-range value:
    try:
        TicketClassification(category="billing", urgency=7, summary="x")
    except ValidationError as e:
        print("out-of-range urgency raises:", e.errors()[0]["msg"])

    # Production-safe pattern: include_raw + check parsing_error, never crash.
    out = model.with_structured_output(TicketClassification, include_raw=True).invoke(
        [HumanMessage(SAMPLE)]
    )
    if out["parsing_error"]:
        print("invalid output -> fall back. error:", out["parsing_error"])
    else:
        print("valid ->", out["parsed"].model_dump())


# ---------------------------------------------------------------------------
# 6. Optional + 'unknown': let the model admit "not enough info"
# ---------------------------------------------------------------------------
def demo_optional_unknown():
    structured = model.with_structured_output(TicketTriage)
    print("clear message ->", structured.invoke([HumanMessage(SAMPLE)]).model_dump())
    print("vague message ->", structured.invoke([HumanMessage(VAGUE)]).model_dump())


# ---------------------------------------------------------------------------
# 7. Nested + lists: classify AND extract in one call
# ---------------------------------------------------------------------------
def demo_nested():
    a = model.with_structured_output(TicketAnalysis).invoke([HumanMessage(SAMPLE)])
    print("category:", a.category, "| urgency:", a.urgency)
    for e in a.entities:
        print(f"  - {e.kind}: {e.text}")


# ---------------------------------------------------------------------------
# 8. (optional) switch the enforcement method if a model is flaky
# ---------------------------------------------------------------------------
def demo_method():
    structured = model.with_structured_output(TicketClassification, method="json_schema")
    print(structured.invoke([HumanMessage(SAMPLE)]).model_dump())


def run(label, fn):
    print(f"\n=== {label} ===")
    try:
        fn()
    except Exception as e:
        print(f"[skipped: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    # Recommended teaching order. Comment out any you don't want to show.
    run("1. basic typed object", demo_basic)
    run("2. descriptions steer the model", demo_descriptions)
    run("3. the typed field IS a router", demo_router)
    run("4. include_raw -> a forced tool call", demo_include_raw)
    run("5. constraints + validation", demo_validation)
    run("6. optional + 'unknown'", demo_optional_unknown)
    run("7. nested extraction", demo_nested)
    # run("8. switch method to json_schema", demo_method)