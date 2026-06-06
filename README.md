# Agents & AI Fundamentals

A hands-on, build-along course that takes you from a hand-written agent loop to
agents you build and control with **LangChain** and **LangGraph**.

> **The arc:** in foundations you *hand-wrote* the agent loop. **Day 1** you learn
> the model / message / tool layer and the easy button (`create_agent`). **Days
> 2–3** you rebuild that loop as a **graph you control**. Everything should land
> as *"oh — this is my raw loop, formalized."*

## Layout

```
.
├── providers.py            # ONE switchboard: Groq (default) / FM Gateway
├── .env.example            # copy to .env, add one key
├── requirements.txt
├── 00_setup/               # Day -1 · get everyone able to call an LLM
│   ├── check_setup.py      #   pre-flight check (green = ready)
│   └── hello_llm.py        #   your first LLM call
└── 01_langchain_fundamentals/   # Day 1 · models, messages, tools, structured output, create_agent
    ├── 01_chat_models_and_messages.py
    ├── 02_tools_and_roundtrip.py
    ├── 03_structured_output.py
    ├── 04_create_agent.py
    ├── 05_mapping.md
    ├── chat_ui.py          #   live streaming chat demo
    ├── exercise.py         #   the daily assignment
    └── exercise_solution.py
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then paste your free Groq key into .env
python 00_setup/check_setup.py
```

Green checks → start [Day −1](00_setup/README.md), then
[Day 1](01_langchain_fundamentals/README.md).

## Switching providers

You never edit example code to change providers. Every script calls
`get_chat_model()` from `providers.py`, which reads `PROVIDER` from your `.env`:

| `PROVIDER=` | Uses | Key needed |
|-------------|------|------------|
| `groq` (default) | `ChatGroq`, free + fast | `GROQ_API_KEY` |
| `fm` | OpenAI client pointed at the FM Gateway `base_url` | `FM_API_KEY` |

## Roadmap

- **Day −1 — Setup** ✅ `00_setup/`
- **Day 1 — LangChain fundamentals** ✅ `01_langchain_fundamentals/`
- **Day 2 — LangGraph: rebuild the loop as a graph you control** *(next)*
- **Day 3 — State, memory & the project** *(next)*
