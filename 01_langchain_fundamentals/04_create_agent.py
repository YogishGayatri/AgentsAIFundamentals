"""
4. create_agent -- the easy button  (~16 min)

Collapse steps 1-4 of 02_tools_and_roundtrip.py into three lines.

Under the hood, create_agent does PRECISELY what you hand-wrote:
  - the agent node calls the model with the messages (after the system prompt)
  - if the AIMessage has tool_calls, the tools node executes them and adds
    ToolMessage objects
  - it calls the model again, repeating until there are no more tool_calls
  - it returns the full list of messages

Read that side-by-side with file 02 -- that's the whole point of the day.

Run:  python 01_langchain_fundamentals/04_create_agent.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from providers import get_chat_model


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    data = {"paris": "Paris: 18°C, light rain",
            "tokyo": "Tokyo: 24°C, clear skies",
            "new york": "New York: 21°C, partly cloudy"}
    return data.get(city.lower(), f"No weather data for {city}.")


model = get_chat_model(temperature=0)

# --- The easy button: the whole loop in three lines -------------------------
agent = create_agent(
    model,                                  # or the string "groq:llama-3.3-70b-versatile"
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Use tools when you need facts.",
)

result = agent.invoke({"messages": [HumanMessage("What's the weather in Paris?")]})
print("Final answer:", result["messages"][-1].content)


# --- Stream it so the room WATCHES the loop turn ----------------------------
print("\n--- streaming the loop (stream_mode='updates') ---")
for chunk in agent.stream(
    {"messages": [HumanMessage("Compare the weather in Tokyo and New York.")]},
    stream_mode="updates",
):
    # You'll see: the model's tool call(s), then the ToolMessage(s), then the answer.
    print(chunk)


# --- Two doors to name but NOT open today -----------------------------------
# (a) Make the agent return a validated object directly -> response_format.
#     This is structured output (file 3) wired straight into the agent.
class WeatherReport(BaseModel):
    location: str = Field(description="The city asked about")
    summary: str = Field(description="Plain-language weather summary")


print("\n--- response_format: agent returns a validated object ---")
typed_agent = create_agent(
    model,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant.",
    response_format=WeatherReport,
)
typed_result = typed_agent.invoke(
    {"messages": [HumanMessage("How's the weather in Tokyo?")]}
)
report = typed_result["structured_response"]   # <- a WeatherReport instance
print("location:", report.location)
print("summary: ", report.summary)

# (b) create_agent(..., checkpointer=InMemorySaver()) + a thread_id gives the
#     agent MEMORY across calls. That's Day 3, so we only name it here.
#
# The honest line that motivates the rest of the week:
#   create_agent HIDES the loop -- great, until you need to CONTROL it (branch,
#   insert a custom step, inspect or resume mid-run). When you need that, you
#   drop to LangGraph. And create_agent is itself built on the LangGraph runtime
#   -- so tomorrow we open up the engine it's already running on.
