"""
chat_ui.py — a live terminal chat so the room WATCHES the agent loop turn.

This is the demo that makes the loop feel real: you type, it shows a "thinking"
beat, then — if it decides it needs a fact — you SEE it reach for a tool, then it
"types" the answer back token by token. It's the same create_agent from file 4,
just streamed into a chat surface instead of printed once.

Run:  python 01_langchain_fundamentals/chat_ui.py
Type 'exit' (or Ctrl-C) to quit.

Why a terminal UI? Zero extra setup for a workshop. If you'd rather have a web
UI, the same agent + streaming code drops straight into a Streamlit app — see the
note at the bottom.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from providers import get_chat_model

# ANSI colors so the trace is readable on stage.
DIM, CYAN, YELLOW, GREEN, RESET = "\033[2m", "\033[96m", "\033[93m", "\033[92m", "\033[0m"


# --- Two tiny generic tools so the agent has a reason to "think" -------------
@tool
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    data = {"paris": "Paris: 18°C, light rain",
            "tokyo": "Tokyo: 24°C, clear skies",
            "new york": "New York: 21°C, partly cloudy",
            "london": "London: 15°C, foggy"}
    return data.get(city.lower(), f"No weather data for {city}.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '3 * (4 + 5)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def build_agent():
    model = get_chat_model(temperature=0)
    return create_agent(
        model,
        tools=[get_weather, calculator],
        system_prompt=(
            "You are a friendly assistant. Use get_weather for weather questions "
            "and calculator for arithmetic. Otherwise just answer directly."
        ),
    )


def typewriter(text: str, delay: float = 0.012):
    """Print text character by character — the 'it's typing' effect."""
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()


def run_turn(agent, history):
    """Stream one agent turn. Show tool use live, then type out the answer."""
    print(f"{DIM}  (thinking…){RESET}", end="\r", flush=True)

    final_text = ""
    # stream_mode='updates' yields one dict per node step: we watch for the
    # model asking to use a tool, and for the tool's result coming back.
    for chunk in agent.stream({"messages": history}, stream_mode="updates"):
        for node, payload in chunk.items():
            for msg in payload.get("messages", []):
                # The model decided to call a tool.
                for call in getattr(msg, "tool_calls", None) or []:
                    print(" " * 20, end="\r")  # clear the "thinking" line
                    print(f"{YELLOW}  🔧 calling {call['name']}({call['args']}){RESET}")
                # A tool returned an observation.
                if type(msg).__name__ == "ToolMessage":
                    print(f"{DIM}  ↳ {msg.content}{RESET}")
                # The model's final natural-language answer.
                if type(msg).__name__ == "AIMessage" and msg.content:
                    final_text = msg.content

    print(" " * 20, end="\r")  # clear any leftover "thinking" line
    print(f"{GREEN}Assistant:{RESET} ", end="", flush=True)
    typewriter(final_text)
    return final_text


def main():
    agent = build_agent()
    history = []

    print(f"\n{CYAN}Live agent chat.{RESET} Try: \"weather in London\", "
          f"\"what is 12 * 9?\", or just chat. Type 'exit' to quit.\n")

    while True:
        try:
            user = input(f"{CYAN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            return
        if not user:
            continue

        history.append(HumanMessage(user))
        answer = run_turn(agent, history)
        # Keep the conversation going by remembering the assistant's reply.
        from langchain_core.messages import AIMessage
        history.append(AIMessage(answer))
        print()


if __name__ == "__main__":
    main()


# ── Want a web UI instead? ──────────────────────────────────────────────────
# The exact same `build_agent()` works in Streamlit. Sketch:
#
#   import streamlit as st
#   agent = build_agent()
#   if prompt := st.chat_input("Ask me anything"):
#       st.chat_message("user").write(prompt)
#       with st.chat_message("assistant"):
#           # st.write_stream(...) over agent.stream(...) gives the typing effect
#           ...
#
# Same model, same tools, same loop — only the surface changes.
