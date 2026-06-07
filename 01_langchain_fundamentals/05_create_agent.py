import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from providers import get_chat_model
 
 
@tool
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    data = {"paris": "Paris: 18C, light rain",
            "tokyo": "Tokyo: 24C, clear skies",
            "new york": "New York: 21C, partly cloudy"}
    return data.get(city.lower(), f"No weather data for {city}.")
 
 
model = get_chat_model(temperature=0)
 

agent = create_agent(
    model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Use tools when you need facts.",
)
 
# Use it: one question.
result = agent.invoke({"messages": [HumanMessage("What's the weather in Paris?")]})
 
print("Answer:", result["messages"][-1].content)
 