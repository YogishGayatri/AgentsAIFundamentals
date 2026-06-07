"""
2. Tools: @tool, bind_tools
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from providers import get_chat_model


#  build a tool
@tool
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    fake_data = {
        "paris": "Paris: 18°C, light rain",
        "tokyo": "Tokyo: 24°C, clear skies",
        "new york": "New York: 21°C, partly cloudy",
    }
    return fake_data.get(city.lower(), f"No weather data for {city}.")



model = get_chat_model(temperature=0)
model_with_tools = model.bind_tools([get_weather])
msgs = [HumanMessage("what 2 multiplied by 3")]
ai = model_with_tools.invoke(msgs)
print("tool_calls:", ai.tool_calls)
msgs.append(ai)
for call in ai.tool_calls:
    if call["name"]== "get_weather":
        result = get_weather.invoke(call["args"])
        msgs.append(ToolMessage(result, tool_call_id=call["id"]))
    else:
        msgs.append(ToolMessage("No tool was called"))

final = model_with_tools.invoke(msgs)
print("Final answer:", final.content)

"""
msgs = [HumanMessage("What's the weather in Tokyo?")]

ai = model_with_tools.invoke(msgs)        # 1. model decides it wants a tool
msgs.append(ai)                           #    remember the request

for call in ai.tool_calls:                # 2. the model only ASKED -- we execute
    result = get_weather.invoke(call["args"])
    msgs.append(ToolMessage(result, tool_call_id=call["id"]))   # 3. feed result back

final = model_with_tools.invoke(msgs)
print("Final answer:", final.content)


"""

