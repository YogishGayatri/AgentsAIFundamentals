"""
Reference solution for exercise.py. Try the exercise yourself first!

Run:  python 01_langchain_fundamentals/exercise_solution.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            "london": "London: 15°C, foggy"}
    return data.get(city.lower(), f"No weather data for {city}.")


@tool
def suggest_clothing(temperature_c: float) -> str:
    """Suggest what to wear for a given temperature in Celsius."""
    if temperature_c < 10:
        return "Cold — wear a warm coat, scarf, and gloves."
    if temperature_c < 20:
        return "Cool — a light jacket or sweater is ideal."
    return "Warm — t-shirt weather, dress light."


class OutfitAdvice(BaseModel):
    city: str = Field(description="The city asked about")
    weather: str = Field(description="The weather summary used")
    recommendation: str = Field(description="A helpful, specific clothing recommendation")


def main():
    model = get_chat_model(temperature=0)

    agent = create_agent(
        model,
        tools=[get_weather, suggest_clothing],
        system_prompt="You are a helpful assistant. Use your tools to give outfit advice.",
        response_format=OutfitAdvice,
    )

    question = "What's the weather in London, and what should I wear?"

    print("--- streaming the trace ---")
    for chunk in agent.stream({"messages": [HumanMessage(question)]}, stream_mode="updates"):
        print(chunk)

    result = agent.invoke({"messages": [HumanMessage(question)]})
    advice = result["structured_response"]
    print("\n--- structured result ---")
    print(f"city:           {advice.city}")
    print(f"weather:        {advice.weather}")
    print(f"recommendation: {advice.recommendation}")


if __name__ == "__main__":
    main()
