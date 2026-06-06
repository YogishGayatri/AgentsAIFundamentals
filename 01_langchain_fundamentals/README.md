# Day 1 · LangChain Fundamentals (~70 min)

> **The arc to keep repeating:** in foundations you *hand-wrote* the agent loop.
> Today you learn the **model / message / tool** layer and the **easy button**
> (`create_agent`). The next two days you rebuild that loop as a **graph you
> control**. Everything today should land as *"oh — this is my raw loop, formalized."*

Every script gets its model from the repo-root `providers.py`, so it runs against
Groq by default and against the FM Gateway by changing one line in `.env`.

Run them in order:

| # | File | Idea | Time |
|---|------|------|------|
| 1 | `01_chat_models_and_messages.py` | The four roles are just Python objects | 12 min |
| 2 | `02_tools_and_roundtrip.py` | `@tool`, `bind_tools`, the tool round-trip = your raw loop | 15 min |
| 3 | `03_structured_output.py` | Make the model return a validated object | 10 min |
| 4 | `04_create_agent.py` | Collapse the loop into 3 lines | 16 min |
| — | `05_mapping.md` | The one-slide "raw loop ↔ LangChain today" table | 4 min |
| — | `chat_app.py` | **Browser UI** (Streamlit): chat + structured-output tabs with a live trace | demo |
| — | `chat_ui.py` | Terminal version of the same live streaming chat | demo |
| — | `exercise.py` | Your hands-on assignment (a TODO skeleton) | 8 min |

```bash
python 01_langchain_fundamentals/01_chat_models_and_messages.py
python 01_langchain_fundamentals/02_tools_and_roundtrip.py
python 01_langchain_fundamentals/03_structured_output.py
python 01_langchain_fundamentals/04_create_agent.py
python 01_langchain_fundamentals/chat_ui.py        # interactive (terminal)
```

### Browser UI (recommended for the workshop)

A Streamlit app that shows the whole Day 1 surface with a **live trace** of what's
happening locally — 🤔 thinking → 🔧 the model asks for a tool → ↳ the tool's
result → the model types the answer. Two tabs: **Agent chat** and **Structured
output**.

```bash
streamlit run 01_langchain_fundamentals/chat_app.py
```

It opens at http://localhost:8501. Pick the provider/model in the sidebar (reads
your `.env`), then try *"weather in London"* or *"what is 12 * 9?"*. Stop with
Ctrl-C in the terminal.

## What we deliberately skip today

So nobody goes hunting for "the rest of LangChain": we are **not** covering deep
LCEL / chain composition, the legacy agents (`AgentExecutor`,
`create_react_agent`, `initialize_agent` — all superseded by `create_agent`), or
the integrations zoo (vector stores, document loaders, retrievers). Those belong
to the RAG world we parked, or are the old way.

Today's surface — **models, messages, tools, structured output, `create_agent`**
— is everything you need to step into LangGraph tomorrow.

## The daily assignment

See `exercise.py`. In short: add a second tool, ask a question that needs
**both** tools, have the agent return a validated object via `response_format`,
then stream it and read the trace.
