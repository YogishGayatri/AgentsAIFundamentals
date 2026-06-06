"""
Day 1 assignment — your hands-on (~8 min in the room, finish at home).

Goal: prove you can wire all four ideas together yourself.

  1. Add a SECOND tool (a skeleton is below — finish it).
  2. Ask a question that needs BOTH tools in one turn.
  3. Have the agent return a validated object via response_format.
  4. Stream it and read the trace.

Look for the  # TODO  markers. A reference solution lives in
exercise_solution.py — try it yourself first!

Run:  python 01_langchain_fundamentals/exercise.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from providers import get_chat_model


# --- Tool 1: given to you ---------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    data = {"paris": "Paris: 18°C, light rain",
            "tokyo": "Tokyo: 24°C, clear skies",
            "london": "London: 15°C, foggy"}
    return data.get(city.lower(), f"No weather data for {city}.")


# --- Tool 2: TODO -----------------------------------------------------------
# Write a tool that returns a short clothing suggestion for a temperature.
# Remember: the DOCSTRING is what the model reads to decide when to call it,
# so make it clear and specific.
@tool
def suggest_clothing(temperature_c: float) -> str:
    """TODO: write a one-line docstring describing what this tool does."""
    # TODO: return e.g. "Bring a warm coat" if cold, "T-shirt weather" if warm.
    raise NotImplementedError("Finish suggest_clothing()")


# --- Structured result: TODO ------------------------------------------------
# Define what a good final answer looks like as a typed object.
class OutfitAdvice(BaseModel):
    city: str = Field(description="The city asked about")
    weather: str = Field(description="The weather summary used")
    # TODO: add a field `recommendation: str` with a helpful description.


def main():
    model = get_chat_model(temperature=0)

    agent = create_agent(
        model,
        tools=[get_weather, suggest_clothing],
        system_prompt="You are a helpful assistant. Use your tools to give outfit advice.",
        response_format=OutfitAdvice,
    )

    # TODO: ask a question that forces BOTH tools (weather -> then clothing).
    question = "What's the weather in London, and what should I wear?"

    print("--- streaming the trace ---")
    for chunk in agent.stream({"messages": [HumanMessage(question)]}, stream_mode="updates"):
        print(chunk)

    result = agent.invoke({"messages": [HumanMessage(question)]})
    advice = result["structured_response"]
    print("\n--- structured result ---")
    print(advice)


if __name__ == "__main__":
    main()
