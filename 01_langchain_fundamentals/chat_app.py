"""
chat_app.py — a Streamlit UI for everything you learned on Day 1.

Why a UI?
---------
The terminal scripts teach the mechanics; this app lets the ROOM watch them.
Every turn shows -- live -- what's happening locally:
    🤔 thinking  ->  🔧 the model asks for a tool  ->  ↳ the tool's result
    ->  the model reads it and types the final answer.

It covers the whole Day 1 surface in two tabs:
    💬 Agent chat       -> messages + tools + create_agent + streaming
    🧱 Structured output -> with_structured_output: text in, a typed object out

Run it (from the repo ROOT, with your venv active):
    streamlit run 01_langchain_fundamentals/chat_app.py

Then your browser opens at http://localhost:8501.
(Stop it with Ctrl-C in the terminal.)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from providers import get_chat_model, DEFAULT_PROVIDER, DEFAULT_MODELS


# ─────────────────────────────────────────────────────────────────────────────
# Tools — the same generic tools from the terminal lessons.
# The docstring is what the model reads to decide when to call each one.
# ─────────────────────────────────────────────────────────────────────────────
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


TOOLS = [get_weather, calculator]


# A typed schema for the structured-output tab (and as a routing-style example).
class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "general"] = Field(
        description="Which queue this message belongs in")
    urgency: int = Field(description="How urgent, 1 (low) to 5 (critical)")
    summary: str = Field(description="One-line summary of what the user wants")


# ─────────────────────────────────────────────────────────────────────────────
# Model / agent builders. Cached so we don't rebuild on every Streamlit rerun.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_agent(provider: str, model_id: str):
    model = get_chat_model(provider=provider, model=model_id, temperature=0)
    return create_agent(
        model,
        tools=TOOLS,
        system_prompt=(
            "You are a friendly assistant. Use get_weather for weather questions "
            "and calculator for arithmetic. Otherwise answer directly and concisely."
        ),
    )


@st.cache_resource(show_spinner=False)
def build_structured_model(provider: str, model_id: str):
    model = get_chat_model(provider=provider, model=model_id, temperature=0)
    return model.with_structured_output(TicketClassification)


def stream_words(text: str, delay: float = 0.02):
    """Yield text word-by-word so st.write_stream gives a 'typing' effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — pick the provider/model and see what's wired up.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LangChain Fundamentals", page_icon="🔗", layout="centered")

with st.sidebar:
    st.header("⚙️ Setup")
    providers = list(DEFAULT_MODELS.keys())          # ["groq", "fm"]
    provider = st.selectbox(
        "Provider", providers,
        index=providers.index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in providers else 0,
        help="Reads keys from your .env. Default comes from PROVIDER in .env.",
    )
    model_id = st.text_input("Model id", value=DEFAULT_MODELS[provider])
    st.caption("Tools available to the agent:")
    st.code("get_weather(city)\ncalculator(expression)", language="text")
    if st.button("🗑️ Clear conversation"):
        st.session_state.pop("display", None)
        st.session_state.pop("lc_history", None)
        st.rerun()

st.title("🔗 LangChain Fundamentals")
st.caption("Watch the agent loop run locally: think → use a tool → answer.")

tab_chat, tab_struct = st.tabs(["💬 Agent chat", "🧱 Structured output"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Agent chat with a live trace of every step.
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    # display: what we re-render on every rerun.  lc_history: real messages for memory.
    st.session_state.setdefault("display", [])
    st.session_state.setdefault("lc_history", [])

    for m in st.session_state["display"]:
        st.chat_message(m["role"]).markdown(m["content"])

    prompt = st.chat_input("Ask me… e.g. 'weather in London' or 'what is 12 * 9?'")
    if prompt:
        st.session_state["display"].append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        st.session_state["lc_history"].append(HumanMessage(prompt))

        with st.chat_message("assistant"):
            final_text = ""
            try:
                agent = build_agent(provider, model_id)
                # st.status is the "what's happening locally" panel.
                with st.status("🤔 Thinking…", expanded=True) as status:
                    for chunk in agent.stream(
                        {"messages": st.session_state["lc_history"]},
                        stream_mode="updates",
                    ):
                        for node, payload in chunk.items():
                            for msg in payload.get("messages", []):
                                # 1. the model ASKS to use a tool
                                for call in getattr(msg, "tool_calls", None) or []:
                                    status.markdown(
                                        f"🔧 **calling `{call['name']}`** with "
                                        f"`{call['args']}`")
                                # 2. the tool's result comes back
                                if type(msg).__name__ == "ToolMessage":
                                    status.markdown(f"↳ result: `{msg.content}`")
                                # 3. the model's final natural-language answer
                                if type(msg).__name__ == "AIMessage" and msg.content:
                                    final_text = msg.content
                    status.update(label="✓ Done", state="complete", expanded=False)

                # Type the answer out, then keep it for memory + redraw.
                st.write_stream(stream_words(final_text))
                st.session_state["lc_history"].append(AIMessage(final_text))
                st.session_state["display"].append(
                    {"role": "assistant", "content": final_text})
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}\n\n"
                         f"Check your .env key for provider '{provider}'.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Structured output: free text in, a validated object out.
# ─────────────────────────────────────────────────────────────────────────────
with tab_struct:
    st.markdown(
        "`with_structured_output(Schema)` forces the model to return a **validated "
        "Pydantic object** instead of free text — something your code can branch on. "
        "Here we triage a support message into a typed `TicketClassification`.")
    example = ("Hi, I was charged twice for my subscription this month and I need "
               "the duplicate refunded before my statement closes tomorrow!")
    text = st.text_area("Incoming message", value=example, height=120)

    if st.button("Classify ➜", type="primary"):
        try:
            structured = build_structured_model(provider, model_id)
            with st.spinner("Calling the model…"):
                result: TicketClassification = structured.invoke([HumanMessage(text)])
            c1, c2 = st.columns(2)
            c1.metric("Category", result.category)
            c2.metric("Urgency", f"{result.urgency} / 5")
            st.markdown(f"**Summary:** {result.summary}")
            with st.expander("The raw typed object (what your code receives)"):
                st.json(result.model_dump())
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}\n\n"
                     f"Check your .env key for provider '{provider}'.")
