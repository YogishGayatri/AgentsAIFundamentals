"""
2. Tools: @tool, bind_tools, and the round-trip  (~15 min)

This is the heart of the day -- it maps one-to-one onto the raw while-loop you
hand-wrote in foundations.

    bind_tools([...])                 "here is the tool menu"
    AIMessage.tool_calls              the model emits a structured INTENTION
    tool.invoke(args)                 YOUR code runs the tool
    ToolMessage(result, tool_call_id) feed the observation back, tied to the call

The only thing separating this from a real agent is wrapping it in a loop --
which is exactly what create_agent (next file) does.

Run:  python 01_langchain_fundamentals/02_tools_and_roundtrip.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from providers import get_chat_model


# --- 1. Build a real tool ---------------------------------------------------
# THE design point of the whole week: the docstring is NOT a comment. It is the
# description the model reads to decide whether and how to call the tool.
# @tool turns the function name + docstring + type hints into the schema the LLM
# sees. Vague docstring -> wrong tool use.
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

# --- 2. Bind the tools so the model can ASK to use one ----------------------
model_with_tools = model.bind_tools([get_weather])

ai = model_with_tools.invoke([HumanMessage("What's the weather in Tokyo?")])
print("tool_calls:", ai.tool_calls)
# -> [{'name': 'get_weather', 'args': {'city': 'Tokyo'}, 'id': 'call_...'}]
# Note: the model did NOT fetch anything. It only emitted an intention.


# --- 3. The full round-trip = your Day-1 loop, in LangChain -----------------
msgs = [HumanMessage("What's the weather in Tokyo?")]

ai = model_with_tools.invoke(msgs)        # 1. model decides it wants a tool
msgs.append(ai)                           #    remember the request

for call in ai.tool_calls:                # 2. the model only ASKED -- we execute
    result = get_weather.invoke(call["args"])
    msgs.append(ToolMessage(result, tool_call_id=call["id"]))   # 3. feed result back

final = model_with_tools.invoke(msgs)     # 4. model reads the result, answers
print("Final answer:", final.content)

# Put this script side by side with create_agent (next file): create_agent is
# literally steps 1-4 above, wrapped in a "while there are still tool_calls" loop.
